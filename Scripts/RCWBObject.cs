using System;
using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    public class RCWBObject : MonoBehaviour
    {
        public Color BasicColor;       // 基础颜色
        
        public float Density;       // 物质密度
        
        [ColorUsage(false, true)] 
        public Color Emission;
        
        [Range(0.0f, 10.0f)]
        [Tooltip("GI系数 (控制全局光照强度，默认1.0)")]
        public float giCoefficient = 1.0f;
        
        private int TextureIndex;   // 如果你有多张图集，这里标记用哪张，单张图集可忽略
        
        private Vector4 uvBox;
    }
}