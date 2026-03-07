using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// PolygonManagerCore 的配置文件
    /// 在 Project 窗口中右键 -> Create -> RadianceCascadesWorldBVH -> Polygon Manager Settings 创建
    /// </summary>
    [CreateAssetMenu(fileName = "PolygonManagerSettings", menuName = "RadianceCascadesWorldBVH/Polygon Manager Settings")]
    public class PolygonManagerSettings : ScriptableObject
    {
        private static PolygonManagerSettings s_Instance;
        
        /// <summary>
        /// 获取全局配置实例（自动从 Resources 文件夹加载）
        /// </summary>
        public static PolygonManagerSettings Instance
        {
            get
            {
                if (s_Instance == null)
                {
                    s_Instance = Resources.Load<PolygonManagerSettings>("PolygonManagerSettings");
                    
                    if (s_Instance == null)
                    {
                        Debug.LogWarning("[PolygonManagerSettings] 未找到配置文件，使用默认值。" +
                            "请在 Assets/Resources 文件夹中创建 PolygonManagerSettings.asset");
                        s_Instance = CreateInstance<PolygonManagerSettings>();
                    }
                }
                return s_Instance;
            }
        }
        
        [Header("场景包围盒")]
        [Tooltip("场景的 AABB 包围盒 (minX, minY, maxX, maxY)")]
        public Vector4 sceneAABB = new Vector4(-100, -100, 100, 100);
        
        [Header("调试设置")]
        [Tooltip("是否在控制台输出调试信息")]
        public bool enableDebugLog = false;
        
        /// <summary>
        /// 将当前配置应用到 PolygonManagerCore
        /// </summary>
        public void ApplyToCore()
        {
            if (PolygonManagerCore.Instance != null)
            {
                PolygonManagerCore.Instance.SceneAABB = sceneAABB;
            }
        }
        
        private void OnValidate()
        {
            // 编辑器中修改时自动应用
            ApplyToCore();
        }
    }
}
