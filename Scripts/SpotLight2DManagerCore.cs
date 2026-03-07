using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.PlayerLoop;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// GPU 端 SpotLight2D 数据结构
    /// 使用 4 个 float4 紧凑打包 (64 bytes)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SpotLight2DGpu
    {
        // float4: 位置 (posX, posY) 和 2D 方向 (dirX, dirY)
        public Vector4 positionDirection;
        
        // float4: 颜色 (R, G, B) 和衰减指数 (falloffExponent)
        public Vector4 colorFalloff;
        
        // float4: 半径和角度（角度存储为 cos 值，便于 GPU 计算）
        // (innerRadius, outerRadius, cosInnerAngle, cosOuterAngle)
        public Vector4 radiiAngles;
        
        // float4: 扩展数据 (height, reserved, reserved, reserved)
        // height: Z 轴高度，用于计算三维光照方向
        public Vector4 heightAndReserved;
    }
    
    /// <summary>
    /// SpotLight2DManagerCore - 基于 PlayerLoop 的 2D 聚光灯管理器
    /// 无需挂载到场景中，游戏启动时自动初始化和运行
    /// </summary>
    public class SpotLight2DManagerCore : IDisposable
    {
        public static SpotLight2DManagerCore Instance { get; private set; }
        
        private static bool s_Initialized = false;
        
        // 注册的聚光灯列表
        private List<SpotLight2D> spotLights = new List<SpotLight2D>();
        
        // GPU 数据
        private List<SpotLight2DGpu> gpuData = new List<SpotLight2DGpu>();
        private ComputeBuffer spotLightBuffer;
        
        // Shader property IDs
        private static readonly int SpotLightBufferID = Shader.PropertyToID("_SpotLight2D_Buffer");
        private static readonly int SpotLightCountID = Shader.PropertyToID("_SpotLight2D_Count");

        #region 静态初始化
        
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void AutoInitialize()
        {
            if (s_Initialized) return;
            
            Instance = new SpotLight2DManagerCore();
            Instance.Initialize();
            
            InjectPlayerLoopUpdate();
            
            Application.quitting += OnApplicationQuit;
            
            s_Initialized = true;
            
            Debug.Log("[SpotLight2DManagerCore] 已自动初始化并注入 PlayerLoop");
        }
        
        private static void InjectPlayerLoopUpdate()
        {
            var playerLoop = PlayerLoop.GetCurrentPlayerLoop();
            
            bool inserted = InsertUpdateSystem(ref playerLoop, typeof(Update), typeof(SpotLight2DManagerCore), OnPlayerLoopUpdate);
            
            if (inserted)
            {
                PlayerLoop.SetPlayerLoop(playerLoop);
            }
            else
            {
                Debug.LogWarning("[SpotLight2DManagerCore] 无法插入 PlayerLoop 更新");
            }
        }
        
        private static bool InsertUpdateSystem(ref PlayerLoopSystem loop, Type targetType, Type newSystemType, PlayerLoopSystem.UpdateFunction updateDelegate)
        {
            if (loop.subSystemList == null)
                return false;
            
            for (int i = 0; i < loop.subSystemList.Length; i++)
            {
                if (loop.subSystemList[i].type == targetType)
                {
                    var subSystems = loop.subSystemList[i].subSystemList;
                    var newSubSystems = new PlayerLoopSystem[(subSystems?.Length ?? 0) + 1];
                    
                    if (subSystems != null)
                    {
                        Array.Copy(subSystems, newSubSystems, subSystems.Length);
                    }
                    
                    newSubSystems[newSubSystems.Length - 1] = new PlayerLoopSystem
                    {
                        type = newSystemType,
                        updateDelegate = updateDelegate
                    };
                    
                    loop.subSystemList[i].subSystemList = newSubSystems;
                    return true;
                }
                
                if (InsertUpdateSystem(ref loop.subSystemList[i], targetType, newSystemType, updateDelegate))
                    return true;
            }
            
            return false;
        }
        
        private static void OnPlayerLoopUpdate()
        {
            Instance?.OnUpdate();
        }
        
        private static void OnApplicationQuit()
        {
            Instance?.Dispose();
            Instance = null;
            s_Initialized = false;
        }
        
        #endregion

        #region 实例方法
        
        private void Initialize()
        {
            // 扫描场景中已存在的 SpotLight2D
            ScanAndRegisterExistingLights();
        }
        
        private void ScanAndRegisterExistingLights()
        {
            SpotLight2D[] existingLights = UnityEngine.Object.FindObjectsByType<SpotLight2D>(FindObjectsSortMode.None);
            foreach (var light in existingLights)
            {
                if (light.isActiveAndEnabled)
                {
                    Register(light);
                }
            }
        }
        
        /// <summary>
        /// 注册一个 SpotLight2D
        /// </summary>
        public void Register(SpotLight2D light)
        {
            if (light == null)
            {
                Debug.LogWarning("SpotLight2DManagerCore.Register: light 为空，跳过注册。");
                return;
            }
            
            if (spotLights.Contains(light))
            {
                return;
            }
            
            spotLights.Add(light);
        }
        
        /// <summary>
        /// 反注册一个 SpotLight2D
        /// </summary>
        public void Unregister(SpotLight2D light)
        {
            if (light == null) return;
            spotLights.Remove(light);
        }
        
        private void OnUpdate()
        {
            // 生成 GPU 数据
            GenerateGpuData();
            
            // 更新 GPU 缓冲区
            UpdateBuffer();
        }
        
        private void GenerateGpuData()
        {
            gpuData.Clear();
            
            for (int i = 0; i < spotLights.Count; i++)
            {
                SpotLight2D light = spotLights[i];
                if (light == null) continue;
                
                Vector2 pos = light.GetPosition();
                Vector2 dir = light.GetDirection();
                Color col = light.color;
                
                // 预计算角度的 cos 值（GPU 端直接用 dot product 比较）
                float cosInner = Mathf.Cos(light.innerAngle * Mathf.Deg2Rad);
                float cosOuter = Mathf.Cos(light.outerAngle * Mathf.Deg2Rad);
                
                SpotLight2DGpu data = new SpotLight2DGpu
                {
                    positionDirection = new Vector4(pos.x, pos.y, dir.x, dir.y),
                    colorFalloff = new Vector4(col.r, col.g, col.b, light.falloffExponent),
                    radiiAngles = new Vector4(light.innerRadius, light.outerRadius, cosInner, cosOuter),
                    heightAndReserved = new Vector4(light.height, 0f, 0f, 0f)
                };
                
                gpuData.Add(data);
            }
        }
        
        private void UpdateBuffer()
        {
            int count = gpuData.Count;
            
            // 设置光源数量（即使为 0 也要设置）
            Shader.SetGlobalInt(SpotLightCountID, count);
            
            if (count == 0)
            {
                return;
            }
            
            // 管理缓冲区大小
            if (spotLightBuffer == null || spotLightBuffer.count < count)
            {
                spotLightBuffer?.Release();
                int newSize = Mathf.Max(16, Mathf.NextPowerOfTwo(count));
                int stride = Marshal.SizeOf<SpotLight2DGpu>();
                spotLightBuffer = new ComputeBuffer(newSize, stride);
            }
            
            // 上传数据
            spotLightBuffer.SetData(gpuData.ToArray(), 0, 0, count);
            
            // 绑定到全局 Shader
            Shader.SetGlobalBuffer(SpotLightBufferID, spotLightBuffer);
        }
        
        public void Dispose()
        {
            spotLightBuffer?.Release();
            spotLightBuffer = null;
            
            Debug.Log("[SpotLight2DManagerCore] 已释放资源");
        }
        
        #endregion

        #region 公开访问器
        
        public int LightCount => spotLights.Count;
        public IReadOnlyList<SpotLight2D> Lights => spotLights;
        
        #endregion
    }
}
