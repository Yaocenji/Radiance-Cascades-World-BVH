#ifndef SPOTLIGHT2D_INC_HLSL
#define SPOTLIGHT2D_INC_HLSL

// 需要先包含 BVH 头文件以使用遮挡检测和兰伯特函数
#include "RCW_BVH_Inc.hlsl"

// SpotLight2D GPU 数据结构 (64 bytes)
// 与 C# 端 SpotLight2DGpu 结构一一对应
struct SpotLight2DData
{
    // (posX, posY, dirX, dirY)
    float4 positionDirection;
    // (R, G, B, falloffExponent)
    float4 colorFalloff;
    // (innerRadius, outerRadius, cosInnerAngle, cosOuterAngle)
    float4 radiiAngles;
    // (height, reserved, reserved, reserved)
    float4 heightAndReserved;
};

// 全局缓冲区和光源数量
StructuredBuffer<SpotLight2DData> _SpotLight2D_Buffer;
int _SpotLight2D_Count;

// =========================================================
// 辅助函数
// =========================================================

float2 GetSpotLightPosition(SpotLight2DData light)
{
    return light.positionDirection.xy;
}

float2 GetSpotLightDirection2D(SpotLight2DData light)
{
    return light.positionDirection.zw;
}

float3 GetSpotLightColor(SpotLight2DData light)
{
    return light.colorFalloff.rgb;
}

float GetSpotLightFalloff(SpotLight2DData light)
{
    return light.colorFalloff.w;
}

float2 GetSpotLightRadii(SpotLight2DData light)
{
    return light.radiiAngles.xy; // (inner, outer)
}

float2 GetSpotLightAngleCos(SpotLight2DData light)
{
    return light.radiiAngles.zw; // (cosInner, cosOuter)
}

float GetSpotLightHeight(SpotLight2DData light)
{
    return light.heightAndReserved.x;
}

/// <summary>
/// 计算从片元指向光源的三维方向（归一化）
/// </summary>
/// <param name="light">光源数据</param>
/// <param name="worldPos">片元世界位置 (2D)</param>
/// <param name="fragmentZ">片元的 Z 坐标（通常为 0）</param>
/// <returns>从片元指向光源的归一化三维方向</returns>
float3 GetLightDirection3D(SpotLight2DData light, float2 worldPos, float fragmentZ)
{
    float2 lightPos2D = GetSpotLightPosition(light);
    float lightHeight = GetSpotLightHeight(light);
    
    float3 lightPos3D = float3(lightPos2D, lightHeight);
    float3 fragPos3D = float3(worldPos, fragmentZ);
    
    return normalize(lightPos3D - fragPos3D);
}

// =========================================================
// 核心光照计算
// =========================================================

/// <summary>
/// 计算 SpotLight2D 对某点的光照贡献（基础版，无遮挡无法线）
/// </summary>
float3 CalculateSpotLight2D_Basic(SpotLight2DData light, float2 worldPos)
{
    float2 lightPos = GetSpotLightPosition(light);
    float2 lightDir = GetSpotLightDirection2D(light);
    float3 lightColor = GetSpotLightColor(light);
    float falloffExp = GetSpotLightFalloff(light);
    
    float innerRadius = light.radiiAngles.x;
    float outerRadius = light.radiiAngles.y;
    float cosInner = light.radiiAngles.z;
    float cosOuter = light.radiiAngles.w;
    
    // 计算到光源的向量
    float2 toLight = lightPos - worldPos;
    float distance = length(toLight);
    
    // 超出外半径，直接返回 0
    if (distance > outerRadius)
        return float3(0, 0, 0);
    
    // 归一化方向（从采样点指向光源）
    float2 toLightDir = toLight / max(distance, 0.0001);
    
    // 计算角度衰减
    float cosAngle = dot(-toLightDir, lightDir);
    
    // 超出外张角，返回 0
    if (cosAngle < cosOuter)
        return float3(0, 0, 0);
    
    // 距离衰减
    float distanceAttenuation = 1.0;
    if (distance > innerRadius)
    {
        float t = (distance - innerRadius) / max(outerRadius - innerRadius, 0.0001);
        distanceAttenuation = pow(1.0 - saturate(t), falloffExp);
    }
    
    // 角度衰减
    float angleAttenuation = 1.0;
    if (cosAngle < cosInner)
    {
        float t = (cosInner - cosAngle) / max(cosInner - cosOuter, 0.0001);
        angleAttenuation = pow(1.0 - saturate(t), falloffExp);
    }
    
    float totalAttenuation = distanceAttenuation * angleAttenuation;
    
    return lightColor * totalAttenuation;
}

/// <summary>
/// 计算 SpotLight2D 对某点的光照贡献（完整版：含遮挡、法线、三维方向）
/// </summary>
/// <param name="light">光源数据</param>
/// <param name="worldPos">片元世界位置 (2D)</param>
/// <param name="normalWS">片元世界空间法线（归一化）</param>
/// <param name="fragmentZ">片元的 Z 坐标（通常为 0）</param>
/// <param name="enableShadow">是否启用阴影/遮挡检测</param>
/// <returns>光照颜色（已应用距离、角度、遮挡、兰伯特衰减）</returns>
float3 CalculateSpotLight2D(SpotLight2DData light, float2 worldPos, float3 normalWS, float fragmentZ, bool enableShadow)
{
    float2 lightPos = GetSpotLightPosition(light);
    float2 lightDir2D = GetSpotLightDirection2D(light);
    float3 lightColor = GetSpotLightColor(light);
    float falloffExp = GetSpotLightFalloff(light);
    
    float innerRadius = light.radiiAngles.x;
    float outerRadius = light.radiiAngles.y;
    float cosInner = light.radiiAngles.z;
    float cosOuter = light.radiiAngles.w;
    
    // 计算到光源的 2D 向量（用于距离和角度计算）
    float2 toLight2D = lightPos - worldPos;
    float distance2D = length(toLight2D);
    
    // 超出外半径，直接返回 0
    if (distance2D > outerRadius)
        return float3(0, 0, 0);
    
    // 归一化 2D 方向
    float2 toLightDir2D = toLight2D / max(distance2D, 0.0001);
    
    // 计算聚光灯角度衰减（2D 平面内）
    float cosAngle = dot(-toLightDir2D, lightDir2D);
    
    // 超出外张角，返回 0
    if (cosAngle < cosOuter)
        return float3(0, 0, 0);
    
    // 距离衰减
    float distanceAttenuation = 1.0;
    if (distance2D > innerRadius)
    {
        float t = (distance2D - innerRadius) / max(outerRadius - innerRadius, 0.0001);
        distanceAttenuation = pow(1.0 - saturate(t), falloffExp);
    }
    
    // 角度衰减
    float angleAttenuation = 1.0;
    if (cosAngle < cosInner)
    {
        float t = (cosInner - cosAngle) / max(cosInner - cosOuter, 0.0001);
        angleAttenuation = pow(1.0 - saturate(t), falloffExp);
    }
    
    // 阴影/遮挡衰减
    float shadowAttenuation = 1.0;
    if (enableShadow)
    {
        shadowAttenuation = CalculateShadowAttenuation(worldPos, lightPos);
    }
    
    // 计算三维光照方向并应用兰伯特
    float3 lightDir3D = GetLightDirection3D(light, worldPos, fragmentZ);
    float lambertAttenuation = CalculateLighting(normalWS, lightDir3D);
    
    // 组合所有衰减
    float totalAttenuation = distanceAttenuation * angleAttenuation * shadowAttenuation * lambertAttenuation;
    
    return lightColor * totalAttenuation;
}

// =========================================================
// 聚合函数
// =========================================================

/// <summary>
/// 计算所有 SpotLight2D 对某点的总光照贡献（基础版）
/// </summary>
float3 CalculateAllSpotLights2D(float2 worldPos)
{
    float3 totalLight = float3(0, 0, 0);
    
    for (int i = 0; i < _SpotLight2D_Count; i++)
    {
        totalLight += CalculateSpotLight2D_Basic(_SpotLight2D_Buffer[i], worldPos);
    }
    
    return totalLight;
}

/// <summary>
/// 计算所有 SpotLight2D 对某点的总光照贡献（完整版：含遮挡、法线）
/// </summary>
/// <param name="worldPos">片元世界位置 (2D)</param>
/// <param name="normalWS">片元世界空间法线（归一化）</param>
/// <param name="fragmentZ">片元的 Z 坐标（通常为 0）</param>
/// <param name="enableShadow">是否启用阴影/遮挡检测</param>
/// <returns>所有光源的总光照颜色</returns>
float3 CalculateAllSpotLights2D(float2 worldPos, float3 normalWS, float fragmentZ, bool enableShadow)
{
    float3 totalLight = float3(0, 0, 0);
    
    for (int i = 0; i < _SpotLight2D_Count; i++)
    {
        totalLight += CalculateSpotLight2D(_SpotLight2D_Buffer[i], worldPos, normalWS, fragmentZ, enableShadow);
    }
    
    return totalLight;
}

/// <summary>
/// 计算所有 SpotLight2D（默认启用阴影，fragmentZ = 0）
/// </summary>
float3 CalculateAllSpotLights2DWithShadow(float2 worldPos, float3 normalWS)
{
    return CalculateAllSpotLights2D(worldPos, normalWS, 0.0, true);
}

// =========================================================
// Sprite 内部光照计算（忽略自阴影）
// =========================================================

/// <summary>
/// 计算 SpotLight2D 对 sprite 内部像素的光照贡献（忽略自阴影）
/// 使用 CalculateShadowAttenuationInterior 来正确处理内部点的遮挡
/// </summary>
/// <param name="light">光源数据</param>
/// <param name="worldPos">片元世界位置 (2D)，位于 sprite 内部</param>
/// <param name="normalWS">片元世界空间法线（归一化）</param>
/// <param name="fragmentZ">片元的 Z 坐标（通常为 0）</param>
/// <returns>光照颜色（已应用距离、角度、遮挡、兰伯特衰减）</returns>
float3 CalculateSpotLight2D_Interior(SpotLight2DData light, float2 worldPos, float3 normalWS, float fragmentZ)
{
    float2 lightPos = GetSpotLightPosition(light);
    float2 lightDir2D = GetSpotLightDirection2D(light);
    float3 lightColor = GetSpotLightColor(light);
    float falloffExp = GetSpotLightFalloff(light);
    
    float innerRadius = light.radiiAngles.x;
    float outerRadius = light.radiiAngles.y;
    float cosInner = light.radiiAngles.z;
    float cosOuter = light.radiiAngles.w;
    
    // 计算到光源的 2D 向量
    float2 toLight2D = lightPos - worldPos;
    float distance2D = length(toLight2D);
    
    // 超出外半径，直接返回 0
    if (distance2D > outerRadius)
        return float3(0, 0, 0);
    
    // 归一化 2D 方向
    float2 toLightDir2D = toLight2D / max(distance2D, 0.0001);
    
    // 计算聚光灯角度衰减（2D 平面内）
    float cosAngle = dot(-toLightDir2D, lightDir2D);
    
    // 超出外张角，返回 0
    if (cosAngle < cosOuter)
        return float3(0, 0, 0);
    
    // 距离衰减
    float distanceAttenuation = 1.0;
    if (distance2D > innerRadius)
    {
        float t = (distance2D - innerRadius) / max(outerRadius - innerRadius, 0.0001);
        distanceAttenuation = pow(1.0 - saturate(t), falloffExp);
    }
    
    // 角度衰减
    float angleAttenuation = 1.0;
    if (cosAngle < cosInner)
    {
        float t = (cosInner - cosAngle) / max(cosInner - cosOuter, 0.0001);
        angleAttenuation = pow(1.0 - saturate(t), falloffExp);
    }
    
    // 阴影/遮挡衰减（使用内部专用版本，忽略自阴影）
    float shadowAttenuation = CalculateShadowAttenuationInterior(worldPos, lightPos);
    
    // 计算三维光照方向并应用兰伯特
    float3 lightDir3D = GetLightDirection3D(light, worldPos, fragmentZ);
    float lambertAttenuation = CalculateLighting(normalWS, lightDir3D);
    
    // 组合所有衰减
    float totalAttenuation = distanceAttenuation * angleAttenuation * shadowAttenuation * lambertAttenuation;
    
    return lightColor * totalAttenuation;
}

/// <summary>
/// 计算所有 SpotLight2D 对 sprite 内部像素的总光照贡献（忽略自阴影）
/// </summary>
/// <param name="worldPos">片元世界位置 (2D)，位于 sprite 内部</param>
/// <param name="normalWS">片元世界空间法线（归一化）</param>
/// <param name="fragmentZ">片元的 Z 坐标（通常为 0）</param>
/// <returns>所有光源的总光照颜色</returns>
float3 CalculateAllSpotLights2D_Interior(float2 worldPos, float3 normalWS, float fragmentZ)
{
    float3 totalLight = float3(0, 0, 0);
    
    for (int i = 0; i < _SpotLight2D_Count; i++)
    {
        totalLight += CalculateSpotLight2D_Interior(_SpotLight2D_Buffer[i], worldPos, normalWS, fragmentZ);
    }
    
    return totalLight;
}

/// <summary>
/// 计算所有 SpotLight2D 对 sprite 内部像素的光照（便捷版，fragmentZ = 0）
/// </summary>
float3 CalculateAllSpotLights2D_Interior(float2 worldPos, float3 normalWS)
{
    return CalculateAllSpotLights2D_Interior(worldPos, normalWS, 0.0);
}

#endif // SPOTLIGHT2D_INC_HLSL
