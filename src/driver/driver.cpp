#include <memory>
#include <sstream>
#include <algorithm>
#include <string>
#include <cstring>
#include <chrono>
#include <cmath>

#ifndef DISABLE_GUI
#include <SDL.h>
#endif

#include "interface.h"
#include "float3.h"
#include "common.h"
#include "image.h"
#include "denoise.h"

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

static constexpr float pi = 3.14159265359f;

struct Camera {
    float3 eye;
    float3 dir;
    float3 right;
    float3 up;
    float w, h;

    Camera(const float3& e, const float3& d, const float3& u, float fov, float ratio) {
        eye = e;
        dir = normalize(d);
        right = normalize(cross(dir, u));
        up = normalize(cross(right, dir));

        w = std::tan(fov * pi / 360.0f);
        h = w / ratio;
    }

    void rotate(float yaw, float pitch) {
        dir = ::rotate(dir, right,  -pitch);
        dir = ::rotate(dir, up,     -yaw);
        dir = normalize(dir);
        right = normalize(cross(dir, up));
        up = normalize(cross(right, dir));
    }

    void move(float x, float y, float z) {
        eye += right * x + up * y + dir * z;
    }
};

void setup_interface(size_t, size_t);
float* get_pixels();
float* get_alb_pixels();
float* get_nrm_pixels();
void clear_pixels();
void cleanup_interface();

#ifndef DISABLE_GUI
static bool handle_events(uint32_t& iter, Camera& cam) {
    static bool camera_on = false;
    static bool arrows[4] = { false, false, false, false };
    static bool speed[2] = { false, false };
    const float rspeed = 0.005f;
    static float tspeed = 0.1f;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        bool key_down = event.type == SDL_KEYDOWN;
        switch (event.type) {
            case SDL_KEYUP:
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:   return true;
                    case SDLK_KP_PLUS:  speed[0] = key_down; break;
                    case SDLK_KP_MINUS: speed[1] = key_down; break;
                    case SDLK_UP:       arrows[0] = key_down; break;
                    case SDLK_DOWN:     arrows[1] = key_down; break;
                    case SDLK_LEFT:     arrows[2] = key_down; break;
                    case SDLK_RIGHT:    arrows[3] = key_down; break;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    SDL_SetRelativeMouseMode(SDL_TRUE);
                    camera_on = true;
                }
                break;
            case SDL_MOUSEBUTTONUP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                    camera_on = false;
                }
                break;
            case SDL_MOUSEMOTION:
                if (camera_on) {
                    cam.rotate(event.motion.xrel * rspeed, event.motion.yrel * rspeed);
                    iter = 0;
                }
                break;
            case SDL_QUIT:
                return true;
            default:
                break;
        }
    }

    if (arrows[0]) cam.move(0, 0,  tspeed);
    if (arrows[1]) cam.move(0, 0, -tspeed);
    if (arrows[2]) cam.move(-tspeed, 0, 0);
    if (arrows[3]) cam.move( tspeed, 0, 0);
    if (arrows[0] | arrows[1] | arrows[2] | arrows[3]) iter = 0;
    if (speed[0]) tspeed *= 1.1f;
    if (speed[1]) tspeed *= 0.9f;
    return false;
}

// gamma corrects the RGB image cointained in data and divides the colors by iter (in place)
static void gamma_correct(size_t width, size_t height, uint32_t iter, float* data, bool doGamma) {
    auto inv_iter = 1.0f / iter;
    auto inv_gamma = doGamma ? 1.0f / 2.2f : 1.0f;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto idx = (y * width + x);
            auto r = data[idx * 3 + 0];
            auto g = data[idx * 3 + 1];
            auto b = data[idx * 3 + 2];

            data[idx * 3 + 0] = clamp(std::pow(r * inv_iter, inv_gamma), 0.0f, 1.0f);
            data[idx * 3 + 1] = clamp(std::pow(g * inv_iter, inv_gamma), 0.0f, 1.0f);
            data[idx * 3 + 2] = clamp(std::pow(b * inv_iter, inv_gamma), 0.0f, 1.0f);
        }
    }
}

static void clamp_image(size_t width, size_t height, float* data) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto idx = (y * width + x);
            auto r = data[idx * 3 + 0];
            auto g = data[idx * 3 + 1];
            auto b = data[idx * 3 + 2];

            data[idx * 3 + 0] = clamp(r, 0.0f, 1.0f);
            data[idx * 3 + 1] = clamp(g, 0.0f, 1.0f);
            data[idx * 3 + 2] = clamp(b, 0.0f, 1.0f);
        }
    }
}

static void update_texture_raw(uint32_t* buf, SDL_Texture* texture, size_t width, size_t height, float* outputPtr) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto r = outputPtr[(y * width + x) * 3 + 0];
            auto g = outputPtr[(y * width + x) * 3 + 1];
            auto b = outputPtr[(y * width + x) * 3 + 2];

            buf[y * width + x] =
                (uint32_t(r * 255.0f) << 16) |
                (uint32_t(g * 255.0f) << 8)  |
                 uint32_t(b * 255.0f);
        }
    }
    SDL_UpdateTexture(texture, nullptr, buf, width * sizeof(uint32_t));
}

static void update_texture(uint32_t* buf, SDL_Texture* texture, size_t width, size_t height, uint32_t iter) {
    auto film = get_pixels();
    auto inv_iter = 1.0f / iter;
    auto inv_gamma = 1.0f / 2.2f;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto r = film[(y * width + x) * 3 + 0];
            auto g = film[(y * width + x) * 3 + 1];
            auto b = film[(y * width + x) * 3 + 2];

            buf[y * width + x] =
                (uint32_t(clamp(std::pow(r * inv_iter, inv_gamma), 0.0f, 1.0f) * 255.0f) << 16) |
                (uint32_t(clamp(std::pow(g * inv_iter, inv_gamma), 0.0f, 1.0f) * 255.0f) << 8)  |
                 uint32_t(clamp(std::pow(b * inv_iter, inv_gamma), 0.0f, 1.0f) * 255.0f);
        }
    }
    SDL_UpdateTexture(texture, nullptr, buf, width * sizeof(uint32_t));
}
#endif
// Saves the RGB image (colors in [0;1]) contained in data
static void save_image(const std::string& out_file, size_t width, size_t height, float* data) {
    ImageRgba32 img;
    img.width = width;
    img.height = height;
    img.pixels.reset(new uint8_t[width * height * 4]);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto r = data[3 * (y * width + x) + 0];
            auto g = data[3 * (y * width + x) + 1];
            auto b = data[3 * (y * width + x) + 2];

            img.pixels[4 * (y * width + x) + 0] = r * 255.0f;
            img.pixels[4 * (y * width + x) + 1] = g * 255.0f;
            img.pixels[4 * (y * width + x) + 2] = b * 255.0f;
            img.pixels[4 * (y * width + x) + 3] = 255;
        }
    }

    if (!save_png(out_file, img))
        error("Failed to save PNG file '", out_file, "'");
}

static inline void check_arg(int argc, char** argv, int arg, int n) {
    if (arg + n >= argc)
        error("Option '", argv[arg], "' expects ", n, " arguments, got ", argc - arg);
}

static inline void usage() {
    std::cout << "Usage: rodent [options]\n"
              << "Available options:\n"
              << "   --help                Shows this message\n"
              << "   --width   pixels      Sets the viewport horizontal dimension (in pixels)\n"
              << "   --height  pixels      Sets the viewport vertical dimension (in pixels)\n"
              << "   --eye     x y z       Sets the position of the camera\n"
              << "   --dir     x y z       Sets the direction vector of the camera\n"
              << "   --up      x y z       Sets the up vector of the camera\n"
              << "   --fov     degrees     Sets the horizontal field of view (in degrees)\n"
              << "   --bench   iterations  Enables benchmarking mode and sets the number of iterations\n"
              << "   --denoise denoise.png Enables denoising with neural network\n"
              << "   --live                Enables live denoising if denoise flag is set\n"
              << "   --aux                 Saves rendered normal and albedo image\n"
              << "   -o        image.png   Writes the output image to a file" << std::endl;
}

int main(int argc, char** argv) {
    std::string out_file;
    size_t bench_iter = 0;
    size_t width  = 1080;
    size_t height = 720;
    float fov = 60.0f;
    float3 eye(0.0f), dir(0.0f, 0.0f, 1.0f), up(0.0f, 1.0f, 0.0f);
    std::string dns = "";
    bool live = false;
    bool aux = false;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (!strcmp(argv[i], "--width")) {
                check_arg(argc, argv, i, 1);
                width = strtoul(argv[++i], nullptr, 10);
            } else if (!strcmp(argv[i], "--height")) {
                check_arg(argc, argv, i, 1);
                height = strtoul(argv[++i], nullptr, 10);
            } else if (!strcmp(argv[i], "--eye")) {
                check_arg(argc, argv, i, 3);
                eye.x = strtof(argv[++i], nullptr);
                eye.y = strtof(argv[++i], nullptr);
                eye.z = strtof(argv[++i], nullptr);
            } else if (!strcmp(argv[i], "--dir")) {
                check_arg(argc, argv, i, 3);
                dir.x = strtof(argv[++i], nullptr);
                dir.y = strtof(argv[++i], nullptr);
                dir.z = strtof(argv[++i], nullptr);
            } else if (!strcmp(argv[i], "--up")) {
                check_arg(argc, argv, i, 3);
                up.x = strtof(argv[++i], nullptr);
                up.y = strtof(argv[++i], nullptr);
                up.z = strtof(argv[++i], nullptr);
            } else if (!strcmp(argv[i], "--fov")) {
                check_arg(argc, argv, i, 1);
                fov = strtof(argv[++i], nullptr);
            } else if (!strcmp(argv[i], "--bench")) {
                check_arg(argc, argv, i, 1);
                bench_iter = strtoul(argv[++i], nullptr, 10);
            } else if (!strcmp(argv[i], "-o")) {
                check_arg(argc, argv, i, 1);
                out_file = argv[++i];
            } else if (!strcmp(argv[i], "--help")) {
                usage();
                return 0;
            } else if (!strcmp(argv[i], "--denoise")) {
                check_arg(argc, argv, i, 1);
                dns = argv[++i];
            } else if (!strcmp(argv[i], "--live")) {
                live = true;
            } else if (!strcmp(argv[i], "--aux")) {
                aux = true;
            } else {
                error("Unknown option '", argv[i], "'");
            }
            continue;
        }
        error("Unexpected argument '", argv[i], "'");
    }
    Camera cam(eye, dir, up, fov, (float)width / (float)height);

#ifdef DISABLE_GUI
    info("Running in console-only mode (compiled with -DDISABLE_GUI).");
    if (bench_iter == 0) {
        warn("Benchmark iterations not set. Defaulting to 1.");
        bench_iter = 1;
    }
#else
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        error("Cannot initialize SDL.");

    auto window = SDL_CreateWindow(
        "Rodent",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        0);
    if (!window)
        error("Cannot create window.");

    auto renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer)
        error("Cannot create renderer.");

    auto texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, width, height);
    if (!texture)
        error("Cannot create texture");

    std::unique_ptr<uint32_t> buf(new uint32_t[width * height]);
#endif

    setup_interface(width, height);

    // Force flush to zero mode for denormals
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    _mm_setcsr(_mm_getcsr() | (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
#endif

    const int img_s = width * height * 3;
    auto spp = get_spp();
    bool done = false;
    uint64_t timing = 0;
    uint32_t frames = 0;
    uint32_t iter = 0;
    std::vector<double> samples_sec;
    live = live && (dns != "");

#ifndef DISABLE_GUI
    anydsl::Array<float> pix, alb, nrm, outputPtr;
    anydsl::Array<float> weights;
    anydsl::Array<uint8_t> memory;
    float* biases;

    if (dns != "") {
        read_in(&weights, &biases);
        memory = anydsl::Array<uint8_t>(get_necessary_mem(width, height));

        pix = anydsl::Array<float>(img_s);
        alb = anydsl::Array<float>(img_s);
        nrm = anydsl::Array<float>(img_s);
        outputPtr = anydsl::Array<float>(img_s);
    }
#endif
    while (!done) {
#ifndef DISABLE_GUI
        done = handle_events(iter, cam);
#endif
        if (iter == 0)
            clear_pixels();

        Settings settings {
            Vec3 { cam.eye.x, cam.eye.y, cam.eye.z },
            Vec3 { cam.dir.x, cam.dir.y, cam.dir.z },
            Vec3 { cam.up.x, cam.up.y, cam.up.z },
            Vec3 { cam.right.x, cam.right.y, cam.right.z },
            cam.w,
            cam.h
        };

        auto ticks = std::chrono::high_resolution_clock::now();
        render(&settings, iter++);
        if (live) {
            anydsl_copy(0, get_pixels(),     0, 0, pix.data(), 0, img_s * sizeof(float));
            anydsl_copy(0, get_alb_pixels(), 0, 0, alb.data(), 0, img_s * sizeof(float));
            anydsl_copy(0, get_nrm_pixels(), 0, 0, nrm.data(), 0, img_s * sizeof(float));

            gamma_correct(width, height, iter, pix.data(), true);
            gamma_correct(width, height, iter, alb.data(), true);
            gamma_correct(width, height, iter, nrm.data(), false);

            denoise(&pix, &alb, &nrm, &memory, &outputPtr, width, height, &weights, biases);

            clamp_image(width, height, outputPtr.data());
        }
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

        if (bench_iter != 0) {
            samples_sec.emplace_back(1000.0 * double(spp * width * height) / double(elapsed_ms));
            if (samples_sec.size() == bench_iter)
                break;
        }

        frames++;
        timing += elapsed_ms;
        if (frames > 10 || timing >= 2500) {
            auto frames_sec = double(frames) * 1000.0 / double(timing);
#ifndef DISABLE_GUI
            std::ostringstream os;
            os << "Rodent [" << frames_sec << " FPS, "
               << iter * spp << " " << "sample" << (iter * spp > 1 ? "s" : "") << "]";
            SDL_SetWindowTitle(window, os.str().c_str());
#endif
            frames = 0;
            timing = 0;
        }

#ifndef DISABLE_GUI
        if(live) {
            update_texture_raw(buf.get(), texture, width, height, outputPtr.data());
        } else {
            update_texture(buf.get(), texture, width, height, iter);
        }
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
#endif
    }

#ifndef DISABLE_GUI

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
#endif

    if (aux || dns != "") {
        gamma_correct(width, height, iter, get_alb_pixels(), true);
        gamma_correct(width, height, iter, get_nrm_pixels(), false);
    }
    if(aux) {
        save_image("albedo.png", width, height, get_alb_pixels());
        save_image("normal.png", width, height, get_nrm_pixels());
        info("Saved auxiliary feature images to albedo.png and normal.png");
    }
    if (out_file != "" || dns != "") {
        gamma_correct(width, height, iter, get_pixels(), true);
        if (out_file != "") {
            save_image(out_file, width, height, get_pixels());
            info("Image saved to '", out_file, "'");
        }
        if(dns != "") {
            anydsl_copy(0, get_pixels(),     0, 0, pix.data(), 0, img_s * sizeof(float));
            anydsl_copy(0, get_alb_pixels(), 0, 0, alb.data(), 0, img_s * sizeof(float));
            anydsl_copy(0, get_nrm_pixels(), 0, 0, nrm.data(), 0, img_s * sizeof(float));

            denoise(&pix, &alb, &nrm, &memory, &outputPtr, width, height, &weights, biases);
            clamp_image(width, height, outputPtr.data());

            save_image("denoised.png", width, height, outputPtr.data());

            info("Denoising done! Saved to '", dns, "'");
        }
    }

    if (dns != "") {
        weights.release();
        memory.release();
        outputPtr.release();
        pix.release();
        nrm.release();
        alb.release();
        free(biases);
    }
    cleanup_interface();

    if (bench_iter != 0) {
        auto inv = 1.0e-6;
        std::sort(samples_sec.begin(), samples_sec.end());
        info("# ", samples_sec.front() * inv,
             "/", samples_sec[samples_sec.size() / 2] * inv,
             "/", samples_sec.back() * inv,
             " (min/med/max Msamples/s)");
    }
    return 0;
}
