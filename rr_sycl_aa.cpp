// Takes the orignal sycl implementation and adds antialisasing through taking multiple samples within the same pixel

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <climits>
#include <chrono>

// Now that I think of it this would be a good place for OOP!!!!!!
// But it would also make SYCL hard :(
enum shape_type
{
    SQUARE,
    TRIANGLE,
    CIRCLE
};

struct colour
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct shape
{
    shape_type type;
    colour colour;
    int x;
    int y;
    int a;
    int b;
    int z_index;
};

struct canvas
{
    int width;
    int height;
    std::vector<shape> objects;
};

bool is_pixel_in_square(const shape &s, float x, float y)
{

    return x >= s.x && x < s.x + s.a && y >= s.y && y < s.y + s.a;
}

bool is_pixel_in_triangle(const shape &s, float x, float y)
{
    if (s.a <= 0 || s.b <= 0)
        return false;

    if (y < s.y || y >= s.y + s.b)
        return false;

    float t = (y - s.y) / static_cast<float>(s.b);

    float half_a = s.a / 2.0f;
    float x_left = s.x + half_a * (1.0f - t);
    float x_right = s.x + half_a * (1.0f + t);

    return x >= x_left && x < x_right;
}

bool is_pixel_in_circle(const shape &s, float x, float y)
{
    if (s.a <= 0)
        return false;

    float radius = s.a / 2.0f;
    float cx = s.x + radius;
    float cy = s.y + radius;
    float dx = x - cx;
    float dy = y - cy;

    return (dx * dx + dy * dy) <= (radius * radius);
}

constexpr int AA_SAMPLES = 4;
constexpr int TOTAL_SAMPLES = AA_SAMPLES * AA_SAMPLES;
void render_canvas(const canvas &c, std::vector<colour> &framebuffer)
{
    sycl::queue queue;
    std::cout << "Render Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::buffer<shape, 1> shape_buffer(const_cast<shape *>(c.objects.data()), c.objects.size());
    sycl::buffer<colour, 1> fb_buffer(framebuffer.data(), framebuffer.size());

    int width = c.width;
    int height = c.height;
    int num_shapes = c.objects.size();

    const auto start = std::chrono::high_resolution_clock::now();

    queue.submit([&](sycl::handler &cgh)
                 {
        auto shapes_acc = shape_buffer.get_access<sycl::access::mode::read>(cgh);
        auto fb_acc = fb_buffer.get_access<sycl::access::mode::write>(cgh); 

        cgh.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx){
            //Get pixel cords
            int py = idx[0];
            int px = idx[1];

            //Find the pixel index in the framebuffer (since its 1D)
            int fb_index = py * width + px;

            int red_sum = 0;
            int green_sum = 0;
            int blue_sum = 0;

            for (int sy = 0; sy < AA_SAMPLES; ++sy) {
                for (int sx = 0; sx < AA_SAMPLES; ++sx) {

                    // Determine the sub pixel location to sample
                    float x = px + (sx + 0.5f) / AA_SAMPLES;
                    float y = py + (sy + 0.5f) / AA_SAMPLES;

                    colour current_colour{255, 255, 255};
                    int current_z = INT_MIN;

                    for (int i = 0; i < num_shapes; ++i) {
                        const auto &s = shapes_acc[i];
                        bool hit = false;

                        switch (s.type) {
                            case SQUARE:
                                hit = is_pixel_in_square(s, x, y);
                                break;

                            case CIRCLE:
                                hit = is_pixel_in_circle(s, x, y);
                                break;

                            case TRIANGLE:
                                hit = is_pixel_in_triangle(s, x, y);
                                break;
                        }

                        if (hit && s.z_index > current_z) {
                            current_colour = s.colour;
                            current_z = s.z_index;
                        }
                    }

                    red_sum += current_colour.r;
                    green_sum += current_colour.g;
                    blue_sum += current_colour.b;
                }
            }   

            colour final_colour{
            static_cast<uint8_t>((red_sum + TOTAL_SAMPLES / 2) / TOTAL_SAMPLES),static_cast<uint8_t>((green_sum + TOTAL_SAMPLES / 2) / TOTAL_SAMPLES),static_cast<uint8_t>((blue_sum + TOTAL_SAMPLES / 2) / TOTAL_SAMPLES)};

            fb_acc[fb_index] = final_colour;

        }); })
        .wait();

    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Parallel work took: " << duration.count() << " microseconds" << std::endl;
}

// Writes a frame buffer to file as a PPM.
// PPM is a very simple uncompressed image file which I can easily write without a libary
// https://en.wikipedia.org/wiki/Netpbm
void save_ppm(const std::string &filename, const std::vector<colour> &fb, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n"
         << width << " " << height << "\n255\n";
    for (const auto &col : fb)
    {
        unsigned char r = static_cast<unsigned char>(col.r);
        unsigned char g = static_cast<unsigned char>(col.g);
        unsigned char b = static_cast<unsigned char>(col.b);
        file.write(reinterpret_cast<char *>(&r), 1);
        file.write(reinterpret_cast<char *>(&g), 1);
        file.write(reinterpret_cast<char *>(&b), 1);
    }
}

int main(int argc, char *argv[])
{

    // Original version hard coded for canvas size
    //  canvas c;
    //  c.width = 1000;
    //  c.height = 1000;

    // // Nested loops to generate a dynamic background pattern
    // for (int y = 0; y < c.height; y += 100)
    // {
    //     for (int x = 0; x < c.width; x += 100)
    //     {
    //         // Alternate shape types based on grid position
    //         shape_type type;
    //         int pattern_selector = (x / 100 + y / 100) % 3;
    //         if (pattern_selector == 0)
    //             type = SQUARE;
    //         else if (pattern_selector == 1)
    //             type = CIRCLE;
    //         else
    //             type = TRIANGLE;

    //         // Generate a smooth color gradient based on coordinates
    //         uint8_t r = static_cast<uint8_t>((x * 255) / c.width);
    //         uint8_t g = static_cast<uint8_t>((y * 255) / c.height);
    //         uint8_t b = static_cast<uint8_t>(180);

    //         c.objects.push_back({type, {r, g, b}, x, y, 80, 80, 1});
    //     }
    // }

    // // Deakin Logo
    // c.objects.push_back({CIRCLE, {255, 255, 255}, 225, 225, 550, 550, 2});
    // c.objects.push_back({SQUARE, {40, 40, 60}, 350, 300, 300, 300, 4});
    // c.objects.push_back({CIRCLE, {40, 40, 60}, 350, 450, 300, 300, 3});

    // int base_x = 387;
    // int base_y = 580;

    // for (int i = 9; i >= 1; --i)
    // {
    //     colour col = {255, 255, 255};

    //     if (i % 2 == 0)
    //     {
    //         col = {40, 40, 60};
    //     }

    //     int size = i * 25;

    //     // Adjust y so the bottom left corners remain aligned
    //     int y = base_y - size;

    //     c.objects.push_back({TRIANGLE, col, base_x, y, size, size, 15 - i});
    // }

    int canvas_size = 1000;

    if (argc == 2)
    {
        canvas_size = std::stoi(argv[1]);

        if (canvas_size <= 0)
        {
            std::cerr << "Canvas size must be positive.\n";
            return 1;
        }
    }

    // Modified version of the original which
    canvas c;
    c.width = canvas_size;
    c.height = canvas_size;

    // Scale relative to the original 1000 x 1000 design
    float scale = c.width / 1000.0f;

    // Fancy background
    int cell_size = c.width / 10;
    int shape_size = static_cast<int>(80 * scale);

    for (int y = 0; y < c.height; y += cell_size)
    {
        for (int x = 0; x < c.width; x += cell_size)
        {
            shape_type type;
            int pattern_selector = (x / cell_size + y / cell_size) % 3;

            if (pattern_selector == 0)
                type = SQUARE;
            else if (pattern_selector == 1)
                type = CIRCLE;
            else
                type = TRIANGLE;

            uint8_t r = static_cast<uint8_t>((x * 255) / c.width);
            uint8_t g = static_cast<uint8_t>((y * 255) / c.height);
            uint8_t b = 180;

            c.objects.push_back({type, {r, g, b}, x, y, shape_size, shape_size, 1});
        }
    }

    // Deakin Logo
    c.objects.push_back({CIRCLE, {255, 255, 255}, static_cast<int>(225 * scale), static_cast<int>(225 * scale), static_cast<int>(550 * scale), static_cast<int>(550 * scale), 2});

    c.objects.push_back({SQUARE, {40, 40, 60}, static_cast<int>(350 * scale), static_cast<int>(300 * scale), static_cast<int>(300 * scale), static_cast<int>(300 * scale), 4});

    c.objects.push_back({CIRCLE, {40, 40, 60}, static_cast<int>(350 * scale), static_cast<int>(450 * scale), static_cast<int>(300 * scale), static_cast<int>(300 * scale), 3});

    // The triangle thing is created by stacking triangles on top of each other with alternating colours
    int base_x = static_cast<int>(387 * scale);
    int base_y = static_cast<int>(580 * scale);

    for (int i = 9; i >= 1; --i)
    {
        colour col = (i % 2 == 0) ? colour{40, 40, 60} : colour{255, 255, 255};

        int size = static_cast<int>(i * 25 * scale);

        // Adjust y so the bottom left corners remain aligned
        int y = base_y - size;

        c.objects.push_back({TRIANGLE, col, base_x, y, size, size, 15 - i});
    }

    // Create the frame buffer for the render to write to
    std::vector<colour> framebuffer(c.width * c.height, {255, 255, 255});

    // Render!
    const auto start = std::chrono::high_resolution_clock::now();
    render_canvas(c, framebuffer);
    const auto stop = std::chrono::high_resolution_clock::now();

    save_ppm("output.ppm", framebuffer, c.width, c.height);
    std::cout << "Rendering complete! Output saved to output.ppm\n";

    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Completed in: " << duration.count() << " microseconds" << std::endl;

    return 0;
}