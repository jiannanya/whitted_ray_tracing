
### A Whitted Style Ray Tracing Implement

Whitted 风格的光线追踪是一种经典的渲染算法，用于生成计算机图形图像。它通过模拟光线从光源发出，经过场景中的物体，最终进入摄像机的过程来渲染图像。这种方法能够实现逼真的反射和折射效果。

**核心概念**

1. 光线 (Ray)：光线由一个起点（Origin）和一个方向（Direction）定义。在光线追踪中，我们从摄像机发射主光线，并从交点发射反射和折射光线。

2. 交点 (Intersection)：光线追踪的核心操作是计算光线与场景中物体的交点。如果光线与多个物体相交，我们只考虑最近的那个交点。

3. 表面法线 (Surface Normal)：在交点处，物体的表面法线对于计算光线的反射和折射方向以及光照非常重要。

4. 材质 (Material)：物体具有不同的材质属性，这些属性决定了光线与物体表面交互的方式。常见的材质类型包括：

   + 漫反射 (Diffuse)：光线在各个方向均匀散射，通常用于模拟粗糙表面。

   + 镜面反射 (Specular)：光线按照反射定律反射，产生高光效果，通常用于模拟光滑表面。

   + 反射 (Reflection)：光线在物体表面完全反射，用于模拟镜子等物体。

   + 折射 (Refraction)：光线穿透物体并改变方向，用于模拟玻璃、水等透明或半透明物体。

5. 光照 (Lighting)：光照模型决定了物体表面如何根据光源和材质属性着色。Whitted 风格光线追踪通常考虑以下光照分量：

   + 环境光 (Ambient Light)：模拟场景中无方向性的背景光。

   + 漫反射光 (Diffuse Light)：由光源直接照射到物体表面并漫反射的光。

   + 镜面反射光 (Specular Light)：由光源直接照射到物体表面并产生高光的光。

   + 阴影 (Shadow)：通过从交点向光源发射阴影光线来判断交点是否被其他物体遮挡。

6. 菲涅尔项 (Fresnel Term)：描述了光线在不同介质界面处反射和折射的能量比例。它取决于入射角和介质的折射率，入射角越大，反射的比例通常越高。

7. 递归追踪 (Recursive Tracing)：Whitted 风格光线追踪通过递归调用 trace_ray 函数来处理反射和折射光线，直到达到最大递归深度或光线不再与物体相交

**伪代码：**

```
function TraceRay(ray, depth):
    if depth > MaxDepth:
        return BlackColor

    Find the closest intersection point (P) and the intersected object (Obj) along the ray.
    if no intersection:
        return BackgroundColor

    Compute normal (N) at intersection point P.
    Compute material properties (Kd, Ks, Shininess, Kr, Kt, IoR) of Obj at P.

    // Ambient and Diffuse-Glossy component
    color = Obj.diffuse_color * AmbientLightColor
    for each light in scene:
        Construct shadow ray from P to light source.
        if shadow ray is not occluded:
            Calculate diffuse component: L_diff = max(0, dot_product(N, LightDirection)) * LightColor * Kd
            Calculate specular component: L_spec = pow(max(0, dot_product(ReflectedLightDirection, ViewDirection)), Shininess) * LightColor * Ks
            color = color + L_diff + L_spec

    // Reflection component
    if Obj has reflection properties (Kr > 0):
        reflection_direction = reflect(ray.direction, N)
        reflection_ray_origin = P + N * epsilon (or P - N * epsilon for inside objects)
        reflected_color = TraceRay(reflection_ray_origin, reflection_direction, depth + 1)
        color = color + reflected_color * Kr

    // Refraction component
    if Obj has refraction properties (Kt > 0):
        refraction_direction = refract(ray.direction, N, IoR)
        refraction_ray_origin = P - N * epsilon (or P + N * epsilon for inside objects)
        refracted_color = TraceRay(refraction_ray_origin, refraction_direction, depth + 1)
        color = color + refracted_color * Kt * (1 - FresnelFactor) // Fresnel factor for energy conservation

    return color

function RenderImage():
    for each pixel (x, y) in image:
        Construct primary ray from camera position through pixel (x, y).
        pixel_color = TraceRay(primary_ray, 0)
        Set pixel (x, y) to pixel_color


```

+ reflect(incident_direction, normal): 计算反射方向。

+ refract(incident_direction, normal, ior): 计算折射方向，其中 ior 是折射率。

+ fresnel(incident_direction, normal, ior): 计算菲涅尔项，用于确定反射和折射光的比例。

+ epsilon: 一个很小的偏移量，用于避免光线与物体自身相交产生的浮点精度问题。

**结果:**

![output picture](render_output.png)