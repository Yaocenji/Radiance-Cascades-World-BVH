using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// 逐场景的 RCWB 设置。挂载到场景中任意 GameObject 即可覆盖全局 PolygonManagerSettings。
    /// 场景中只应存在一个实例；若有多个，取第一个找到的。
    /// </summary>
    public class RCWBSceneSettings : MonoBehaviour
    {
        [Tooltip("本场景的 BVH 包围盒（世界空间 xMin, yMin, xMax, yMax）。覆盖全局 Settings 中的 sceneAABB。")]
        public Vector4 sceneAABB = new Vector4(-100, -100, 100, 100);

        private void OnEnable()
        {
            Apply();
        }

        private void OnValidate()
        {
            if (Application.isPlaying && PolygonManagerCore.Instance != null)
                Apply();
        }

        private void Apply()
        {
            PolygonManagerCore.EnsureInitialized();
            if (PolygonManagerCore.Instance != null)
                PolygonManagerCore.Instance.SceneAABB = sceneAABB;
        }
    }
}
