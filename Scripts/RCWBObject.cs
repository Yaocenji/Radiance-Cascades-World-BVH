using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace RadianceCascadesWorldBVH
{
    public class RCWBObject : MonoBehaviour
    {
        [Tooltip("是否作为墙体参与 Polygon BVH 构建")]
        public bool IsWall = true;

        [Tooltip("自定义轮廓 Profile（优先于 Sprite 物理形状）。留空则退回到 Sprite 物理形状。")]
        public RCWBContourProfile ContourProfile;

        public Color BasicColor;       // 基础颜色

        public float Density;       // 物质密度

        [ColorUsage(false, true)]
        public Color Emission;

        [Range(0.0f, 10.0f)]
        [Tooltip("GI系数 (控制全局光照强度，默认1.0)")]
        public float giCoefficient = 1.0f;

        private int TextureIndex;   // 如果你有多张图集，这里标记用哪张，单张图集可忽略

        private Vector4 uvMatrix;       // 2x2线性变换矩阵，存储为(m00, m01, m10, m11)
        private Vector2 uvTranslation;  // 平移向量

        private SpriteRenderer spriteRenderer;
        private Renderer activeRenderer;        // 通用渲染器引用（SpriteRenderer 或 TilemapRenderer）
        private bool isTile;                    // 是否为 Tilemap 类型

        private MaterialPropertyBlock mpb;
        private static readonly int RotationSinCosID = Shader.PropertyToID("_RotationSinCos");
        private static readonly int EmissionID = Shader.PropertyToID("_Emission");
        private static readonly int GICoefficientID = Shader.PropertyToID("_GICoefficient");

        public Vector4 UVMatrix => uvMatrix;
        public Vector2 UVTranslation => uvTranslation;
        public bool IsTile => isTile;

        private void Reset()
        {
            // 编辑器中添加组件时自动检测：Tilemap 类型默认 IsWall = false
            if (GetComponent<TilemapRenderer>() != null)
            {
                IsWall = false;
            }
        }

        private void Awake()
        {
            spriteRenderer = GetComponent<SpriteRenderer>();

            if (spriteRenderer != null)
            {
                activeRenderer = spriteRenderer;
                isTile = false;
            }
            else
            {
                var tilemapRenderer = GetComponent<TilemapRenderer>();
                if (tilemapRenderer != null)
                {
                    activeRenderer = tilemapRenderer;
                    isTile = true;
                }
            }

            mpb = new MaterialPropertyBlock();
        }

        private void OnEnable()
        {
            if (activeRenderer == null)
            {
                spriteRenderer = GetComponent<SpriteRenderer>();
                if (spriteRenderer != null)
                {
                    activeRenderer = spriteRenderer;
                    isTile = false;
                }
                else
                {
                    var tilemapRenderer = GetComponent<TilemapRenderer>();
                    if (tilemapRenderer != null)
                    {
                        activeRenderer = tilemapRenderer;
                        isTile = true;
                    }
                }
            }

            // 非墙体不参与 Polygon BVH 构建
            if (!IsWall) return;

            // 优先向 PolygonManagerCore 注册（PlayerLoop 驱动，无需场景挂载）
            PolygonManagerCore.EnsureInitialized();

            if (PolygonManagerCore.Instance != null)
            {
                PolygonManagerCore.Instance.Register(this, spriteRenderer);
            }
            // 兼容旧的 PolygonManager（场景挂载模式）
            else if (PolygonManager.Instance != null)
            {
                PolygonManager.Instance.Register(this, spriteRenderer);
            }
        }

        private void OnDisable()
        {
            // 优先从 PolygonManagerCore 反注册
            if (PolygonManagerCore.Instance != null)
            {
                PolygonManagerCore.Instance.Unregister(this);
            }
            // 兼容旧的 PolygonManager
            else if (PolygonManager.Instance != null)
            {
                PolygonManager.Instance.Unregister(this);
            }
        }

        private void Update()
        {
            if (!isTile)
                ComputeWorldToAtlasUVTransform();
            UpdateMaterialPropertyBlock();
        }

        private void UpdateMaterialPropertyBlock()
        {
            if (activeRenderer == null) return;

            // 先获取当前的 MPB（保留其他脚本或 Unity 内部设置的属性）
            activeRenderer.GetPropertyBlock(mpb);

            // 计算 z 轴旋转的 cos 和 sin
            float rotationZ = -1 * transform.eulerAngles.z * Mathf.Deg2Rad;
            float cosZ = Mathf.Cos(rotationZ);
            float sinZ = Mathf.Sin(rotationZ);

            // 设置旋转属性 (x = cos, y = sin)
            mpb.SetVector(RotationSinCosID, new Vector4(cosZ, sinZ, 0f, 0f));

            // 设置自发光属性
            mpb.SetColor(EmissionID, Emission);

            // 设置gi系数
            mpb.SetFloat(GICoefficientID, giCoefficient);

            // 应用回渲染器
            activeRenderer.SetPropertyBlock(mpb);
        }
        
        /// <summary>
        /// 计算从2D世界空间到Atlas UV空间的仿射变换
        /// UV_atlas = uvMatrix * worldPos + uvTranslation
        /// 其中uvMatrix是2x2矩阵，按行优先存储为Vector4(m00, m01, m10, m11)
        /// </summary>
        public void ComputeWorldToAtlasUVTransform()
        {
            if (spriteRenderer == null)
                spriteRenderer = GetComponent<SpriteRenderer>();
            
            Sprite sprite = spriteRenderer?.sprite;
            // 必须检查 texture 是否存在，以及 textureRect 是否有效
            if (sprite == null || sprite.texture == null)
            {
                uvMatrix = new Vector4(1, 0, 0, 1);
                uvTranslation = Vector2.zero;
                return;
            }
            
            // --- Step 1: 世界空间 -> 本地空间 (保持不变) ---
            Vector3 worldPos = transform.position;
            Vector3 worldScale = transform.lossyScale;
            float rotationZ = transform.eulerAngles.z * Mathf.Deg2Rad;
            
            float flipX = spriteRenderer.flipX ? -1f : 1f;
            float flipY = spriteRenderer.flipY ? -1f : 1f;
            
            float cosTheta = Mathf.Cos(-rotationZ);
            float sinTheta = Mathf.Sin(-rotationZ);
            
            // 防止除零
            float invScaleX = Mathf.Abs(worldScale.x) > 1e-6f ? 1f / (worldScale.x * flipX) : 0f;
            float invScaleY = Mathf.Abs(worldScale.y) > 1e-6f ? 1f / (worldScale.y * flipY) : 0f;
            
            float m00_wl = invScaleX * cosTheta;
            float m01_wl = -invScaleX * sinTheta;
            float m10_wl = invScaleY * sinTheta;
            float m11_wl = invScaleY * cosTheta;
            
            float tx_wl = -(m00_wl * worldPos.x + m01_wl * worldPos.y);
            float ty_wl = -(m10_wl * worldPos.x + m11_wl * worldPos.y);

            // --- Step 2: 本地空间 -> Atlas UV (核心修正) ---
            // 不再依赖 Bounds，而是直接使用 PPU 和 Offset 计算
            
            // Atlas 的总宽高
            float atlasWidth = sprite.texture.width;
            float atlasHeight = sprite.texture.height;
            
            // Sprite 的 PPU (决定了缩放比例)
            float ppu = sprite.pixelsPerUnit;
            
            // Sprite 在 Atlas 中的矩形区域
            Rect texRect = sprite.textureRect;
            
            // 关键：裁剪偏移量。如果 Unity 裁剪了左边的透明像素，这个值会记录裁剪了多少。
            // 它的含义是：Atlas 中图像的左下角，相对于原始未裁剪图像左下角的偏移。
            Vector2 texRectOffset = sprite.textureRectOffset;
            
            // 原始 Pivot (相对于未裁剪图像左下角的像素位置)
            Vector2 pivot = sprite.pivot;

            // 推导：
            // 1. LocalPos * PPU = 距离 Pivot 的像素距离
            // 2. + Pivot = 在原始未裁剪图片中的像素坐标 (以左下角为原点)
            // 3. - texRectOffset = 在 Atlas 切片中的局部像素坐标 (处理裁剪)
            // 4. + texRect.position = 在 Atlas 大图中的绝对像素坐标
            // 5. / atlasSize = UV
            
            // 缩放因子：PPU / AtlasSize
            float scaleX_la = ppu / atlasWidth;
            float scaleY_la = ppu / atlasHeight;
            
            // 平移因子：(Pivot - Offset + AtlasRectPos) / AtlasSize
            // 注意：因为我们是把 Local(0,0) 映射过去，而 Local(0,0) 就是 Pivot 点
            float transX_la = (pivot.x - texRectOffset.x + texRect.x) / atlasWidth;
            float transY_la = (pivot.y - texRectOffset.y + texRect.y) / atlasHeight;

            // --- Step 3: 组合矩阵 ---
            // UV = M_la * (M_wl * P + T_wl) + T_la
            //    = (M_la * M_wl) * P + (M_la * T_wl + T_la)
            
            uvMatrix = new Vector4(
                scaleX_la * m00_wl,
                scaleX_la * m01_wl,
                scaleY_la * m10_wl,
                scaleY_la * m11_wl
            );
            
            uvTranslation = new Vector2(
                scaleX_la * tx_wl + transX_la,
                scaleY_la * ty_wl + transY_la
            );
        }
        
        private void OnDrawGizmos()
        {
            if (!IsWall) return;
            DrawContourGizmos(new Color(0.2f, 0.8f, 0.2f, 0.4f));
        }

        private void OnDrawGizmosSelected()
        {
            if (!IsWall) return;
            DrawContourGizmos(new Color(0.8f, 0.1f, 0.8f, 1.0f));
        }

        private void DrawContourGizmos(Color color)
        {
            Gizmos.color = color;

            if (ContourProfile != null && ContourProfile.IsValid())
            {
                // 使用 ContourProfile 的局部空间点，经 transform 转到世界空间
                foreach (ContourLoopData loop in ContourProfile.Loops)
                {
                    if (!loop.IsValid()) continue;

                    IReadOnlyList<Vector2> pts = loop.PointsLocal;
                    int count = pts.Count;
                    int edgeCount = loop.Closed ? count : count - 1;

                    for (int i = 0; i < edgeCount; i++)
                    {
                        Vector3 a = transform.TransformPoint(new Vector3(pts[i].x, pts[i].y, 0f));
                        Vector3 b = transform.TransformPoint(new Vector3(pts[(i + 1) % count].x, pts[(i + 1) % count].y, 0f));
                        Gizmos.DrawLine(a, b);
                    }
                }
            }
            else if (spriteRenderer != null && spriteRenderer.sprite != null)
            {
                // 回退：绘制 Sprite 物理形状
                Sprite spr = spriteRenderer.sprite;
                int shapeCount = spr.GetPhysicsShapeCount();
                var shapePoints = new List<Vector2>();

                for (int s = 0; s < shapeCount; s++)
                {
                    shapePoints.Clear();
                    int ptCount = spr.GetPhysicsShape(s, shapePoints);

                    for (int i = 0; i < ptCount; i++)
                    {
                        Vector3 a = transform.TransformPoint(shapePoints[i]);
                        Vector3 b = transform.TransformPoint(shapePoints[(i + 1) % ptCount]);
                        Gizmos.DrawLine(a, b);
                    }
                }
            }
        }

        /// <summary>
        /// 使用计算好的变换，将世界空间点转换为Atlas UV坐标（用于调试验证）
        /// </summary>
        public Vector2 WorldToAtlasUV(Vector2 worldPoint)
        {
            return new Vector2(
                uvMatrix.x * worldPoint.x + uvMatrix.y * worldPoint.y + uvTranslation.x,
                uvMatrix.z * worldPoint.x + uvMatrix.w * worldPoint.y + uvTranslation.y
            );
        }
    }
}
