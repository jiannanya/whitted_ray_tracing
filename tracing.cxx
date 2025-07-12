#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

constexpr float kFloatMax = std::numeric_limits<float>::max();

inline float clamp(const float& min_value, const float& max_value,
                   const float& input_value) {
  return std::max(min_value, std::min(max_value, input_value));
}

inline bool solve_quadratic_equation(const float& a, const float& b,
                                     const float& c, float& root0,
                                     float& root1) {
  float discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
    return false;
  else if (discriminant == 0)
    root0 = root1 = -0.5f * b / a;
  else {
    float q = (b > 0) ? -0.5f * (b + sqrtf(discriminant))
                      : -0.5f * (b - sqrtf(discriminant));
    root0 = q / a;
    root1 = c / q;
  }
  if (root0 > root1) std::swap(root0, root1);
  return true;
}

enum class MaterialType {
  kDiffuseAndGlossy,
  kReflectionAndRefraction,
  kReflection
};

inline float generate_random_float() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

inline void print_progress(float progress) {
  std::cout << "Progress: " << static_cast<int>(progress * 100.0) << "%\r";
  std::cout.flush();
}

class Vector3f {
 public:
  float x, y, z;

  Vector3f() : x(0), y(0), z(0) {}
  Vector3f(float val) : x(val), y(val), z(val) {}
  Vector3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}

  Vector3f operator*(const float& r) const {
    return Vector3f(x * r, y * r, z * r);
  }
  Vector3f operator/(const float& r) const {
    return Vector3f(x / r, y / r, z / r);
  }
  Vector3f operator*(const Vector3f& v) const {
    return Vector3f(x * v.x, y * v.y, z * v.z);
  }
  Vector3f operator-(const Vector3f& v) const {
    return Vector3f(x - v.x, y - v.y, z - v.z);
  }
  Vector3f operator+(const Vector3f& v) const {
    return Vector3f(x + v.x, y + v.y, z + v.z);
  }
  Vector3f operator-() const { return Vector3f(-x, -y, -z); }

  Vector3f& operator+=(const Vector3f& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  friend Vector3f operator*(const float& r, const Vector3f& v) {
    return Vector3f(v.x * r, v.y * r, v.z * r);
  }
  friend std::ostream& operator<<(std::ostream& os, const Vector3f& v) {
    return os << v.x << ", " << v.y << ", " << v.z;
  }
};

class Vector2f {
 public:
  float x, y;

  Vector2f() : x(0), y(0) {}
  Vector2f(float val) : x(val), y(val) {}
  Vector2f(float xx, float yy) : x(xx), y(yy) {}

  Vector2f operator*(const float& r) const { return Vector2f(x * r, y * r); }
  Vector2f operator+(const Vector2f& v) const {
    return Vector2f(x + v.x, y + v.y);
  }
};

inline Vector3f lerp(const Vector3f& a, const Vector3f& b, const float& t) {
  return a * (1 - t) + b * t;
}

inline Vector3f normalize(const Vector3f& v) {
  float magnitude_squared = v.x * v.x + v.y * v.y + v.z * v.z;
  if (magnitude_squared > 0) {
    float inverse_magnitude = 1 / sqrtf(magnitude_squared);
    return Vector3f(v.x * inverse_magnitude, v.y * inverse_magnitude,
                    v.z * inverse_magnitude);
  }
  return v;
}

inline float dot_product(const Vector3f& a, const Vector3f& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vector3f cross_product(const Vector3f& a, const Vector3f& b) {
  return Vector3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

class Object {
 public:
  MaterialType material_type;
  float index_of_refraction;
  float diffuse_coefficient, specular_coefficient;
  Vector3f diffuse_color;
  float specular_exponent;

  Object()
      : material_type(MaterialType::kDiffuseAndGlossy),
        index_of_refraction(1.3f),
        diffuse_coefficient(0.8f),
        specular_coefficient(0.2f),
        diffuse_color(0.2f),
        specular_exponent(25.0f) {}

  virtual ~Object() = default;

  virtual bool intersect(const Vector3f& origin, const Vector3f& direction,
                         float& t_near, uint32_t& index,
                         Vector2f& uv) const = 0;

  virtual void get_surface_properties(const Vector3f& hit_point,
                                      const Vector3f& view_direction,
                                      const uint32_t& index, const Vector2f& uv,
                                      Vector3f& normal, Vector2f& st) const = 0;

  virtual Vector3f eval_diffuse_color(const Vector2f& st) const {
    return diffuse_color;
  }
};

class Light {
 public:
  Vector3f position;
  Vector3f intensity;

  Light(const Vector3f& p, const Vector3f& i) : position(p), intensity(i) {}
  virtual ~Light() = default;
};

class Sphere : public Object {
 public:
  Vector3f center;
  float radius, radius_squared;

  Sphere(const Vector3f& c, const float& r)
      : center(c), radius(r), radius_squared(r * r) {}

  bool intersect(const Vector3f& origin, const Vector3f& direction,
                 float& t_near, uint32_t&, Vector2f&) const override {
    Vector3f L = origin - center;
    float a = dot_product(direction, direction);
    float b = 2 * dot_product(direction, L);
    float c = dot_product(L, L) - radius_squared;
    float t0, t1;
    if (!solve_quadratic_equation(a, b, c, t0, t1)) return false;
    if (t0 < 0) t0 = t1;
    if (t0 < 0) return false;
    t_near = t0;
    return true;
  }

  void get_surface_properties(const Vector3f& P, const Vector3f&,
                              const uint32_t&, const Vector2f&, Vector3f& N,
                              Vector2f&) const override {
    N = normalize(P - center);
  }
};

inline bool ray_triangle_intersect(const Vector3f& v0, const Vector3f& v1,
                                   const Vector3f& v2, const Vector3f& origin,
                                   const Vector3f& direction, float& t_near,
                                   float& u, float& v) {
  Vector3f edge1 = v1 - v0;
  Vector3f edge2 = v2 - v0;
  Vector3f pvec = cross_product(direction, edge2);
  float det = dot_product(edge1, pvec);
  if (det == 0) return false;
  float inv_det = 1 / det;
  Vector3f tvec = origin - v0;
  u = dot_product(tvec, pvec) * inv_det;
  if (u < 0 || u > 1) return false;
  Vector3f qvec = cross_product(tvec, edge1);
  v = dot_product(direction, qvec) * inv_det;
  if (v < 0 || u + v > 1) return false;
  t_near = dot_product(edge2, qvec) * inv_det;
  return t_near > 0;
}

class MeshTriangle : public Object {
 public:
  std::unique_ptr<Vector3f[]> vertices;
  uint32_t num_triangles;
  std::unique_ptr<uint32_t[]> vertex_indices;
  std::unique_ptr<Vector2f[]> st_coordinates;

  MeshTriangle(const Vector3f* verts, const uint32_t* vert_indices,
               const uint32_t& num_tris, const Vector2f* st) {
    uint32_t max_index = 0;
    for (uint32_t i = 0; i < num_tris * 3; ++i) {
      if (vert_indices[i] > max_index) max_index = vert_indices[i];
    }
    max_index += 1;

    vertices = std::make_unique<Vector3f[]>(max_index);
    std::copy(verts, verts + max_index, vertices.get());

    vertex_indices = std::make_unique<uint32_t[]>(num_tris * 3);
    std::copy(vert_indices, vert_indices + (num_tris * 3),
              vertex_indices.get());

    num_triangles = num_tris;

    st_coordinates = std::make_unique<Vector2f[]>(max_index);
    std::copy(st, st + max_index, st_coordinates.get());
  }

  bool intersect(const Vector3f& origin, const Vector3f& direction,
                 float& t_near, uint32_t& index, Vector2f& uv) const override {
    bool intersected = false;
    t_near = kFloatMax;
    for (uint32_t k = 0; k < num_triangles; ++k) {
      const Vector3f& v0 = vertices[vertex_indices[k * 3]];
      const Vector3f& v1 = vertices[vertex_indices[k * 3 + 1]];
      const Vector3f& v2 = vertices[vertex_indices[k * 3 + 2]];
      float t, u_coord, v_coord;
      if (ray_triangle_intersect(v0, v1, v2, origin, direction, t, u_coord,
                                 v_coord) &&
          t < t_near) {
        t_near = t;
        uv.x = u_coord;
        uv.y = v_coord;
        index = k;
        intersected = true;
      }
    }
    return intersected;
  }

  void get_surface_properties(const Vector3f&, const Vector3f&,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& normal, Vector2f& st) const override {
    const Vector3f& v0 = vertices[vertex_indices[index * 3]];
    const Vector3f& v1 = vertices[vertex_indices[index * 3 + 1]];
    const Vector3f& v2 = vertices[vertex_indices[index * 3 + 2]];
    Vector3f edge0 = normalize(v1 - v0);
    Vector3f edge1 = normalize(v2 - v1);
    normal = normalize(cross_product(edge0, edge1));
    const Vector2f& st0 = st_coordinates[vertex_indices[index * 3]];
    const Vector2f& st1 = st_coordinates[vertex_indices[index * 3 + 1]];
    const Vector2f& st2 = st_coordinates[vertex_indices[index * 3 + 2]];
    st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
  }

  Vector3f eval_diffuse_color(const Vector2f& st) const override {
    float scale = 5.0f;
    float pattern_x = fmodf(st.x * scale, 1.0f);
    float pattern_y = fmodf(st.y * scale, 1.0f);
    float pattern = (pattern_x > 0.5f) ^ (pattern_y > 0.5f);
    return lerp(Vector3f(0.815f, 0.235f, 0.031f),
                Vector3f(0.937f, 0.937f, 0.231f), pattern);
  }
};

class Scene {
 public:
  int width;
  int height;
  double field_of_view;
  Vector3f background_color;
  int max_recursion_depth;
  float shadow_epsilon;

  Scene(int w, int h)
      : width(w),
        height(h),
        field_of_view(90.0),
        background_color(0.235294f, 0.67451f, 0.843137f),
        max_recursion_depth(5),
        shadow_epsilon(0.00001f) {}

  void add_object(std::unique_ptr<Object> object) {
    objects_.push_back(std::move(object));
  }

  void add_light(std::unique_ptr<Light> light) {
    lights_.push_back(std::move(light));
  }

  const std::vector<std::unique_ptr<Object>>& get_objects() const {
    return objects_;
  }

  const std::vector<std::unique_ptr<Light>>& get_lights() const {
    return lights_;
  }

 private:
  std::vector<std::unique_ptr<Object>> objects_;
  std::vector<std::unique_ptr<Light>> lights_;
};

struct HitInfo {
  float t_near;
  uint32_t index;
  Vector2f uv;
  Object* hit_object;
};

class RayTracer {
 public:
  void render_image(const Scene& scene);

 private:
  inline float degrees_to_radians(const float& degrees) const;

  Vector3f compute_reflection(const Vector3f& incident,
                              const Vector3f& normal) const;

  Vector3f compute_refraction(const Vector3f& incident, const Vector3f& normal,
                              const float& ior) const;

  float compute_fresnel(const Vector3f& incident, const Vector3f& normal,
                        const float& ior) const;

  std::optional<HitInfo> find_intersection(
      const Vector3f& origin, const Vector3f& direction,
      const std::vector<std::unique_ptr<Object>>& objects) const;

  Vector3f trace_ray(const Vector3f& origin, const Vector3f& direction,
                     const Scene& scene, int depth) const;
};

inline float RayTracer::degrees_to_radians(const float& degrees) const {
  return degrees * std::numbers::pi_v<float> / 180.0f;
}

Vector3f RayTracer::compute_reflection(const Vector3f& incident,
                                       const Vector3f& normal) const {
  return incident - 2 * dot_product(incident, normal) * normal;
}

Vector3f RayTracer::compute_refraction(const Vector3f& incident,
                                       const Vector3f& normal,
                                       const float& ior) const {
  float cosi = clamp(-1.0f, 1.0f, dot_product(incident, normal));
  float etai = 1.0f, etat = ior;
  Vector3f n = normal;
  if (cosi < 0) {
    cosi = -cosi;
  } else {
    std::swap(etai, etat);
    n = -normal;
  }
  float eta = etai / etat;
  float k = 1 - eta * eta * (1 - cosi * cosi);
  return k < 0 ? Vector3f(0.0f) : eta * incident + (eta * cosi - sqrtf(k)) * n;
}

float RayTracer::compute_fresnel(const Vector3f& incident,
                                 const Vector3f& normal,
                                 const float& ior) const {
  float cosi = clamp(-1.0f, 1.0f, dot_product(incident, normal));
  float etai = 1.0f, etat = ior;
  if (cosi > 0) std::swap(etai, etat);
  float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
  if (sint >= 1)
    return 1.0f;
  else {
    float cost = sqrtf(std::max(0.f, 1 - sint * sint));
    cosi = fabsf(cosi);
    float Rs =
        ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp =
        ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etai * cost));
    return (Rs * Rs + Rp * Rp) / 2.0f;
  }
}

std::optional<HitInfo> RayTracer::find_intersection(
    const Vector3f& origin, const Vector3f& direction,
    const std::vector<std::unique_ptr<Object>>& objects) const {
  float closest_t = kFloatMax;
  std::optional<HitInfo> closest_hit;
  for (const auto& object : objects) {
    float t = kFloatMax;
    uint32_t idx;
    Vector2f uv;
    if (object->intersect(origin, direction, t, idx, uv) && t < closest_t) {
      closest_hit.emplace();
      closest_hit->hit_object = object.get();
      closest_hit->t_near = t;
      closest_hit->index = idx;
      closest_hit->uv = uv;
      closest_t = t;
    }
  }
  return closest_hit;
}

Vector3f RayTracer::trace_ray(const Vector3f& origin, const Vector3f& direction,
                              const Scene& scene, int depth) const {
  if (depth > scene.max_recursion_depth) return Vector3f(0.0f);

  Vector3f pixel_color = scene.background_color;
  if (auto hit = find_intersection(origin, direction, scene.get_objects());
      hit) {
    Vector3f hit_point = origin + direction * hit->t_near;
    Vector3f normal;
    Vector2f tex_coords;
    hit->hit_object->get_surface_properties(hit_point, direction, hit->index,
                                            hit->uv, normal, tex_coords);

    switch (hit->hit_object->material_type) {
      case MaterialType::kReflectionAndRefraction: {
        Vector3f reflect_dir = normalize(compute_reflection(direction, normal));
        Vector3f refract_dir = normalize(compute_refraction(
            direction, normal, hit->hit_object->index_of_refraction));
        Vector3f reflect_origin =
            (dot_product(reflect_dir, normal) < 0)
                ? hit_point - normal * scene.shadow_epsilon
                : hit_point + normal * scene.shadow_epsilon;
        Vector3f refract_origin =
            (dot_product(refract_dir, normal) < 0)
                ? hit_point - normal * scene.shadow_epsilon
                : hit_point + normal * scene.shadow_epsilon;
        Vector3f reflect_color =
            trace_ray(reflect_origin, reflect_dir, scene, depth + 1);
        Vector3f refract_color =
            trace_ray(refract_origin, refract_dir, scene, depth + 1);
        float kr = compute_fresnel(direction, normal,
                                   hit->hit_object->index_of_refraction);
        pixel_color = reflect_color * kr + refract_color * (1.0f - kr);
        break;
      }
      case MaterialType::kReflection: {
        float kr = compute_fresnel(direction, normal,
                                   hit->hit_object->index_of_refraction);
        Vector3f reflect_dir = compute_reflection(direction, normal);
        Vector3f reflect_origin =
            (dot_product(reflect_dir, normal) < 0)
                ? hit_point + normal * scene.shadow_epsilon
                : hit_point - normal * scene.shadow_epsilon;
        pixel_color =
            trace_ray(reflect_origin, reflect_dir, scene, depth + 1) * kr;
        break;
      }
      default: {  // MaterialType::kDiffuseAndGlossy
        Vector3f diffuse_sum(0.0f);
        Vector3f specular_sum(0.0f);
        Vector3f shadow_origin =
            (dot_product(direction, normal) < 0)
                ? hit_point + normal * scene.shadow_epsilon
                : hit_point - normal * scene.shadow_epsilon;
        for (auto& light : scene.get_lights()) {
          Vector3f light_dir = light->position - hit_point;
          float light_dist2 = dot_product(light_dir, light_dir);
          light_dir = normalize(light_dir);
          float NdotL = std::max(0.f, dot_product(light_dir, normal));
          auto shadow_hit =
              find_intersection(shadow_origin, light_dir, scene.get_objects());
          bool in_shadow =
              shadow_hit &&
              (shadow_hit->t_near * shadow_hit->t_near < light_dist2);
          diffuse_sum += in_shadow ? Vector3f(0.0f) : light->intensity * NdotL;
          Vector3f reflect_dir = compute_reflection(-light_dir, normal);
          specular_sum +=
              powf(std::max(0.f, -dot_product(reflect_dir, direction)),
                   hit->hit_object->specular_exponent) *
              light->intensity;
        }
        pixel_color = diffuse_sum *
                          hit->hit_object->eval_diffuse_color(tex_coords) *
                          hit->hit_object->diffuse_coefficient +
                      specular_sum * hit->hit_object->specular_coefficient;
        break;
      }
    }
  }
  return pixel_color;
}

void RayTracer::render_image(const Scene& scene) {
  std::vector<Vector3f> framebuffer(scene.width * scene.height);
  float scale = std::tan(degrees_to_radians(scene.field_of_view * 0.5f));
  float aspect_ratio =
      static_cast<float>(scene.width) / static_cast<float>(scene.height);
  Vector3f camera_pos(0.0f);
  int pixel_idx = 0;

  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      float px = (2.0f * (static_cast<float>(x) + 0.5f) / scene.width - 1.0f) *
                 scale * aspect_ratio;
      float py =
          (1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / scene.height) * scale;
      Vector3f ray_dir = normalize(Vector3f(px, py, -1.0f));
      framebuffer[pixel_idx++] = trace_ray(camera_pos, ray_dir, scene, 0);
    }
    print_progress(static_cast<float>(y) / scene.height);
  }

  std::ofstream ofs("render_output.ppm", std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "Error: Could not open file for writing." << std::endl;
    return;
  }
  ofs << "P6\n" << scene.width << " " << scene.height << "\n255\n";
  for (int i = 0; i < scene.height * scene.width; ++i) {
    unsigned char color[3];
    color[0] =
        static_cast<unsigned char>(255 * clamp(0.0f, 1.0f, framebuffer[i].x));
    color[1] =
        static_cast<unsigned char>(255 * clamp(0.0f, 1.0f, framebuffer[i].y));
    color[2] =
        static_cast<unsigned char>(255 * clamp(0.0f, 1.0f, framebuffer[i].z));
    ofs.write(reinterpret_cast<char*>(color), 3);
  }
  ofs.close();
  std::cout << "\nRendering complete. Image saved to render_output.ppm\n";
}

Scene create_sample_scene(int width, int height) {
  Scene scene(width, height);
  auto sphere1 = std::make_unique<Sphere>(Vector3f(-1.0f, 0.0f, -12.0f), 2.0f);
  sphere1->material_type = MaterialType::kDiffuseAndGlossy;
  sphere1->diffuse_color = Vector3f(0.6f, 0.7f, 0.8f);
  scene.add_object(std::move(sphere1));

  auto sphere2 = std::make_unique<Sphere>(Vector3f(0.5f, -0.5f, -8.0f), 1.5f);
  sphere2->index_of_refraction = 1.5f;
  sphere2->material_type = MaterialType::kReflectionAndRefraction;
  scene.add_object(std::move(sphere2));

  Vector3f plane_vertices[4] = {{-5.0f, -3.0f, -6.0f},
                                {5.0f, -3.0f, -6.0f},
                                {5.0f, -3.0f, -16.0f},
                                {-5.0f, -3.0f, -16.0f}};
  uint32_t plane_indices[6] = {0, 1, 3, 1, 2, 3};
  Vector2f plane_st_coords[4] = {
      {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  auto plane = std::make_unique<MeshTriangle>(plane_vertices, plane_indices, 2,
                                              plane_st_coords);
  plane->material_type = MaterialType::kDiffuseAndGlossy;
  scene.add_object(std::move(plane));

  scene.add_light(
      std::make_unique<Light>(Vector3f(-20.0f, 70.0f, 20.0f), Vector3f(0.5f)));
  scene.add_light(
      std::make_unique<Light>(Vector3f(30.0f, 50.0f, -12.0f), Vector3f(0.5f)));

  return scene;
}

void render_scene(Scene& scene) {
  RayTracer tracer;
  tracer.render_image(scene);
}

int main() {
  constexpr int kImageWidth = 500;
  constexpr int kImageHeight = 500;
  Scene scene = create_sample_scene(kImageWidth, kImageHeight);
  render_scene(scene);
  return 0;
}