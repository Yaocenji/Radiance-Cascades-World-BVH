using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.PlayerLoop;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// PolygonManagerCore - 基于 PlayerLoop 的 BVH 管理器
    /// 无需挂载到场景中，游戏启动时自动初始化和运行
    /// </summary>
    public class PolygonManagerCore : IDisposable
    {
        public static PolygonManagerCore Instance { get; private set; }
        
        // 是否已初始化
        private static bool s_Initialized = false;
        
        // 要参与 BVH 的物体（通过注册机制自动管理）
        private List<SpriteRenderer> spriteRenderers = new List<SpriteRenderer>();
        private List<RCWBObject> rcwObjects = new List<RCWBObject>();
        
        // 场景包围盒（可通过 Settings 配置）
        public Vector4 SceneAABB { get; set; } = new Vector4(-100, -100, 100, 100);
        
        // 用于 BVH 的边集合
        private List<edgeBVH> edges = new List<edgeBVH>();
        
        // BVH 构建器
        private PolygonBVHConstructor bvhConstructor;
        private PolygonBVHConstructorAccelerated bvhConstructorAccelerated;
        
        // 材质数据（与 rcwObjects 完全同序）
        private List<MaterialData> materialData = new List<MaterialData>();
        
        // ComputeBuffer
        private ComputeBuffer edgeBuffer;
        private ComputeBuffer nodeBuffer;
        private ComputeBuffer gpuNodeEdgeBuffer;
        private ComputeBuffer materialBuffer;
        
        // 是否已完成首次初始化（BVH构建器创建）
        private bool m_BvhInitialized = false;

        #region 静态初始化
        
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetStatics()
        {
            Instance = null;
            s_Initialized = false;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void AutoInitialize()
        {
            EnsureInitialized();
            return;
            
            Instance = new PolygonManagerCore();
            Instance.Initialize();
            
            // 注入到 PlayerLoop
            InjectPlayerLoopUpdate();
            
            // 订阅应用退出事件
            Application.quitting += OnApplicationQuit;
            
            s_Initialized = true;
            
            Debug.Log("[PolygonManagerCore] 已自动初始化并注入 PlayerLoop");
        }
        
        public static void EnsureInitialized()
        {
            if (s_Initialized && Instance != null) return;
            
            Instance = new PolygonManagerCore();
            Instance.Initialize();
            
            // 娉ㄥ叆鍒?PlayerLoop
            InjectPlayerLoopUpdate();
            
            // 璁㈤槄搴旂敤閫€鍑轰簨浠?
            Application.quitting -= OnApplicationQuit;
            Application.quitting += OnApplicationQuit;
            
            s_Initialized = true;
            
            Debug.Log("[PolygonManagerCore] 宸茶嚜鍔ㄥ垵濮嬪寲骞舵敞鍏?PlayerLoop");
        }

        private static void InjectPlayerLoopUpdate()
        {
            var playerLoop = PlayerLoop.GetCurrentPlayerLoop();
            
            // 在 Update 阶段之后插入我们的更新
            bool inserted = InsertUpdateSystem(ref playerLoop, typeof(Update), typeof(PolygonManagerCore), OnPlayerLoopUpdate);
            
            if (inserted)
            {
                PlayerLoop.SetPlayerLoop(playerLoop);
            }
            else
            {
                Debug.LogWarning("[PolygonManagerCore] 无法插入 PlayerLoop 更新，回退到备用方案");
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
                    // 在目标系统的子系统列表末尾添加我们的更新
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
                
                // 递归查找
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
            // 加载配置
            LoadSettings();
            
            // 扫描场景中已存在的 RCWBObject
            ScanAndRegisterExistingObjects();
        }
        
        /// <summary>
        /// 从 PolygonManagerSettings 加载配置
        /// </summary>
        private void LoadSettings()
        {
            var settings = PolygonManagerSettings.Instance;
            if (settings != null)
            {
                SceneAABB = settings.sceneAABB;
            }
        }
        
        /// <summary>
        /// 扫描场景中所有活跃的 RCWBObject 并注册
        /// </summary>
        private void ScanAndRegisterExistingObjects()
        {
            RCWBObject[] existingObjects = UnityEngine.Object.FindObjectsByType<RCWBObject>(FindObjectsSortMode.None);
            foreach (var obj in existingObjects)
            {
                if (obj.isActiveAndEnabled)
                {
                    SpriteRenderer sr = obj.GetComponent<SpriteRenderer>();
                    if (sr != null)
                    {
                        Register(obj, sr);
                    }
                }
            }
        }
        
        /// <summary>
        /// 注册一个 RCWBObject 及其对应的 SpriteRenderer
        /// </summary>
        public void Register(RCWBObject rcwbObject, SpriteRenderer spriteRenderer)
        {
            if (rcwbObject == null || spriteRenderer == null)
            {
                Debug.LogWarning("PolygonManagerCore.Register: rcwbObject 或 spriteRenderer 为空，跳过注册。");
                return;
            }
            
            if (rcwObjects.Contains(rcwbObject))
            {
                return;
            }
            
            rcwObjects.Add(rcwbObject);
            spriteRenderers.Add(spriteRenderer);
        }
        
        /// <summary>
        /// 反注册一个 RCWBObject
        /// </summary>
        public void Unregister(RCWBObject rcwbObject)
        {
            if (rcwbObject == null) return;
            
            int index = rcwObjects.IndexOf(rcwbObject);
            if (index >= 0)
            {
                rcwObjects.RemoveAt(index);
                spriteRenderers.RemoveAt(index);
            }
        }
        
        private void OnUpdate()
        {
            // 延迟初始化 BVH 构建器（确保有注册的对象后再创建）
            if (!m_BvhInitialized)
            {
                bvhConstructor = new PolygonBVHConstructor(edges, spriteRenderers);
                bvhConstructorAccelerated = new PolygonBVHConstructorAccelerated(edges, spriteRenderers);
                m_BvhInitialized = true;
            }
            
            // 绑定 Atlas
            if (spriteRenderers.Count > 0 && spriteRenderers[0] != null && spriteRenderers[0].sprite != null)
            {
                BindAtlasGlobal(spriteRenderers[0].sprite);
            }
            
            // 重新生成材质数据
            GenerateMaterialData();
            
            // 构建 BVH
            bvhConstructorAccelerated.GetBvhEdges();
            bvhConstructorAccelerated.CalculateMortonCodes(SceneAABB);
            bvhConstructorAccelerated.SortMortonCodes();
            bvhConstructorAccelerated.BuildBVHStructure();
            bvhConstructorAccelerated.RefitBVH();
            bvhConstructorAccelerated.ReorderBVHToBFS();
            bvhConstructorAccelerated.PackGpuNodes();
            
            // 更新 GPU 缓冲区
            UpdateBuffers(edges, bvhConstructorAccelerated.nodes);
        }
        
        private void GenerateMaterialData()
        {
            materialData.Clear();
            
            for (int i = 0; i < rcwObjects.Count; i++)
            {
                RCWBObject obj = rcwObjects[i];
                if (obj == null) continue;
                
                MaterialData mat = new MaterialData();
                mat.BasicColor = obj.BasicColor;
                mat.Density = obj.Density;
                mat.Emission = obj.Emission;
                mat.TextureIndex = 0;
                mat.uvMatrix = obj.UVMatrix;
                mat.uvTranslation = obj.UVTranslation;
                materialData.Add(mat);
            }
        }
        
        public void BindAtlasGlobal(Sprite anySpriteInAtlas)
        {
            if (anySpriteInAtlas == null || anySpriteInAtlas.texture == null) return;
            
            Texture2D mainAtlas = anySpriteInAtlas.texture;
            Shader.SetGlobalTexture("_RCWB_Atlas", mainAtlas);
        }
        
        public void UpdateBuffers(List<edgeBVH> edges, LBVHNodeRaw[] nodes)
        {
            // 管理 GPU Node Edge Buffer
            int gpuNodeCount = bvhConstructorAccelerated.GpuNodeCount;
            if (gpuNodeCount > 0)
            {
                if (gpuNodeEdgeBuffer == null || gpuNodeEdgeBuffer.count < gpuNodeCount)
                {
                    gpuNodeEdgeBuffer?.Release();
                    int newSize = Mathf.NextPowerOfTwo(gpuNodeCount);
                    int s = Marshal.SizeOf<LBVHNodeGpu>();
                    gpuNodeEdgeBuffer = new ComputeBuffer(newSize, s);
                }
                gpuNodeEdgeBuffer.SetData(bvhConstructorAccelerated.gpuNodes, 0, 0, gpuNodeCount);
            }
            
            // 管理 Material Buffer
            if (materialData.Count > 0)
            {
                if (materialBuffer == null || materialBuffer.count < materialData.Count)
                {
                    materialBuffer?.Release();
                    int newSize = Mathf.CeilToInt(materialData.Count * 1.2f);
                    int s = Marshal.SizeOf<MaterialData>();
                    materialBuffer = new ComputeBuffer(newSize, s);
                }
                materialBuffer.SetData(materialData.ToArray(), 0, 0, materialData.Count);
            }
            
            // 绑定到全局 Shader
            if (gpuNodeEdgeBuffer != null)
                Shader.SetGlobalBuffer("_BVH_NodeEdge_Buffer", gpuNodeEdgeBuffer);
            if (materialBuffer != null)
                Shader.SetGlobalBuffer("_BVH_Material_Buffer", materialBuffer);
            Shader.SetGlobalInt("_BVH_Root_Index", bvhConstructorAccelerated.rootNodeIndex);
        }
        
        public void Dispose()
        {
            edgeBuffer?.Release();
            nodeBuffer?.Release();
            gpuNodeEdgeBuffer?.Release();
            materialBuffer?.Release();
            bvhConstructorAccelerated?.Dispose();
            
            edgeBuffer = null;
            nodeBuffer = null;
            gpuNodeEdgeBuffer = null;
            materialBuffer = null;
            
            Debug.Log("[PolygonManagerCore] 已释放资源");
        }
        
        #endregion

        #region 公开访问器（用于调试）
        
        public List<edgeBVH> Edges => edges;
        public List<MaterialData> Materials => materialData;
        public PolygonBVHConstructorAccelerated BvhConstructor => bvhConstructorAccelerated;
        public int RegisteredObjectCount => rcwObjects.Count;
        
        #endregion
    }
}
