using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// 2D 聚光灯组件
    /// 以 transform 中心为原点，transform.right 为中心方向
    /// </summary>
    public class SpotLight2D : MonoBehaviour
    {
        [Header("颜色")]
        [ColorUsage(false, true)]
        [Tooltip("HDR 光源颜色")]
        public Color color = Color.white;
        
        [Header("距离衰减")]
        [Min(0f)]
        [Tooltip("内半径：在此范围内光强为 100%")]
        public float innerRadius = 1f;
        
        [Min(0f)]
        [Tooltip("外半径：超过此范围光强为 0%")]
        public float outerRadius = 5f;
        
        [Header("角度衰减")]
        [Range(0f, 180f)]
        [Tooltip("内张角（度）：在此角度内光强为 100%")]
        public float innerAngle = 15f;
        
        [Range(0f, 180f)]
        [Tooltip("外张角（度）：超过此角度光强为 0%")]
        public float outerAngle = 45f;
        
        [Header("衰减曲线")]
        [Range(0.01f, 10f)]
        [Tooltip("衰减指数：控制内外之间的衰减曲线 (1=线性, <1=快衰减, >1=慢衰减)")]
        public float falloffExponent = 1f;
        
        [Header("3D 高度")]
        [Tooltip("光源的 Z 轴高度（用于计算三维光照方向）")]
        public float height = 1f;

        private void OnEnable()
        {
            // 向 SpotLight2DManagerCore 注册
            if (SpotLight2DManagerCore.Instance != null)
            {
                SpotLight2DManagerCore.Instance.Register(this);
            }
        }
        
        private void OnDisable()
        {
            // 从 SpotLight2DManagerCore 反注册
            if (SpotLight2DManagerCore.Instance != null)
            {
                SpotLight2DManagerCore.Instance.Unregister(this);
            }
        }

        private void OnValidate()
        {
            // 确保内半径不超过外半径
            if (innerRadius > outerRadius)
                innerRadius = outerRadius;
            
            // 确保内张角不超过外张角
            if (innerAngle > outerAngle)
                innerAngle = outerAngle;
        }

        /// <summary>
        /// 获取世界空间位置 (2D)
        /// </summary>
        public Vector2 GetPosition()
        {
            return transform.position;
        }
        
        /// <summary>
        /// 获取光源方向 (归一化的 transform.right)
        /// </summary>
        public Vector2 GetDirection()
        {
            return transform.right;
        }

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            Vector3 pos = transform.position;
            Vector2 dir = transform.right;
            
            // 绘制方向
            Gizmos.color = color;
            Gizmos.DrawRay(pos, dir * outerRadius);
            
            // 绘制内外半径圆弧
            DrawArc(pos, dir, innerRadius, innerAngle, Color.green);
            DrawArc(pos, dir, outerRadius, outerAngle, Color.yellow);
            
            // 绘制内外张角边界线
            float innerAngleRad = innerAngle * Mathf.Deg2Rad;
            float outerAngleRad = outerAngle * Mathf.Deg2Rad;
            
            Gizmos.color = Color.green;
            Vector2 innerEdge1 = RotateVector(dir, innerAngleRad) * innerRadius;
            Vector2 innerEdge2 = RotateVector(dir, -innerAngleRad) * innerRadius;
            Gizmos.DrawLine(pos, pos + (Vector3)innerEdge1);
            Gizmos.DrawLine(pos, pos + (Vector3)innerEdge2);
            
            Gizmos.color = Color.yellow;
            Vector2 outerEdge1 = RotateVector(dir, outerAngleRad) * outerRadius;
            Vector2 outerEdge2 = RotateVector(dir, -outerAngleRad) * outerRadius;
            Gizmos.DrawLine(pos, pos + (Vector3)outerEdge1);
            Gizmos.DrawLine(pos, pos + (Vector3)outerEdge2);
        }
        
        private void DrawArc(Vector3 center, Vector2 direction, float radius, float angle, Color color)
        {
            Gizmos.color = color;
            int segments = 32;
            float angleRad = angle * Mathf.Deg2Rad;
            float step = (angleRad * 2f) / segments;
            
            Vector3 prevPoint = center + (Vector3)(RotateVector(direction, -angleRad) * radius);
            for (int i = 1; i <= segments; i++)
            {
                float currentAngle = -angleRad + step * i;
                Vector3 currentPoint = center + (Vector3)(RotateVector(direction, currentAngle) * radius);
                Gizmos.DrawLine(prevPoint, currentPoint);
                prevPoint = currentPoint;
            }
        }
        
        private Vector2 RotateVector(Vector2 v, float angle)
        {
            float cos = Mathf.Cos(angle);
            float sin = Mathf.Sin(angle);
            return new Vector2(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
        }
#endif
    }
}
