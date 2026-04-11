# RadianceCascadesWorldBVH 系统文档

> 面向开发者与 AI 上下文快速检索。描述轮廓管理 + BVH 构建 + GPU 上传的完整流程。

---

## 1. 系统概述

本系统为 2D 场景中的不透明物体提取几何轮廓，构建 LBVH（Linear BVH），上传到 GPU，供光照/遮蔽 Compute Shader 使用。

**核心特性：**
- `PolygonManagerCore`：PlayerLoop 驱动，无需在场景中挂载 GameObject，游戏启动时自动初始化
- 每个场景物体挂载 `RCWBObject`，注册后自动参与每帧 BVH 构建
- 轮廓来源优先级：`RCWBContourProfile`（自定义，每物体独立）> Sprite 物理形状（fallback）
- BVH 构建使用 Burst + Job System 加速（`PolygonBVHConstructorAccelerated`）

---

## 2. 文件结构

```
RadianceCascadesWorldBVH/
├── Scripts/
│   ├── RCWBObject.cs                     # 场景物体组件（注册/反注册，Gizmos）
│   ├── RCWBContourProfile.cs             # ScriptableObject：存储每物体的自定义轮廓
│   ├── PolygonManagerCore.cs             # 主管理器（PlayerLoop，单例，非 MonoBehaviour）
│   ├── PolygonManagerSettings.cs         # 配置资产（SceneAABB、Atlas、Profile 输出目录）
│   ├── PolygonBVHConstructorAccelerated.cs  # Burst/Job BVH 构建器（实际使用）
│   ├── PolygonBVHConstructor.cs          # 纯 C# BVH 构建器（已弃用，仅保留参考）
│   └── PolygonManager.cs                 # [OBSOLETE] 旧 MonoBehaviour 管理器，已废弃
├── Editor/
│   ├── RCWBContourProfileGenerator.cs    # 编辑器工具：从 Sprite 像素生成 ContourProfile
│   ├── SpritePhysicsShapeGenerator.cs    # 编辑器工具（Legacy）：生成 Sprite 物理形状
│   └── RCWBContourProfileEditor.cs       # ContourProfile 的自定义 Inspector
├── RCWB_Asmd.asmdef                      # 主程序集（引用 Burst/Collections/URP）
└── Editor/RCWB_Editor_Asmd.asmdef        # Editor 程序集（引用主程序集 + Unity.2D.Sprite.Editor）
```

---

## 3. 完整数据流

```
[场景 GameObject]
       │ OnEnable
       ▼
RCWBObject.OnEnable()
  └─ PolygonManagerCore.EnsureInitialized()   ← 总是成功（PlayerLoop 驱动）
  └─ PolygonManagerCore.Instance.Register(rcwObj, spriteRenderer)
       │ 保存到 List<RCWBObject> rcwObjects
       │         List<SpriteRenderer> spriteRenderers   （两表严格同序）

═══════════════════════ 每帧 PlayerLoop (Update 末尾) ═══════════════════════

PolygonManagerCore.OnUpdate()
  │
  ├─ [1] Shader.SetGlobalTexture("_RCWB_Atlas", ...)
  │        优先 PolygonManagerSettings.atlasTexture，否则取 spriteRenderers[0].sprite.texture
  │
  ├─ [2] GenerateMaterialData()
  │        遍历 rcwObjects → MaterialData { BasicColor, Emission, Density, uvMatrix, uvTranslation }
  │
  ├─ [3] bvhConstructorAccelerated.GetBvhEdges()
  │        for each (rcwObj, spriteRenderer) 对：
  │          if rcwObj.ContourProfile 有效：
  │            ContourLoopData.pointsLocal (局部 Unity 单位)
  │              → transform.TransformPoint() → 世界坐标 edgeBVH
  │          else (fallback)：
  │            sprite.GetPhysicsShape() (已是局部 Unity 单位)
  │              → transform.TransformPoint() → 世界坐标 edgeBVH
  │        edgeBVH { start:Vector2, end:Vector2, matIdx:int }
  │
  ├─ [4] CalculateMortonCodes(SceneAABB)   [Burst 并行]
  ├─ [5] SortMortonCodes()                  [NativeSort]
  ├─ [6] BuildBVHStructure()               [Burst 并行，Karras 2012]
  ├─ [7] RefitBVH()                         [Burst 并行，自底向上 AABB]
  ├─ [8] ReorderBVHToBFS()                  [BFS 重排，串行]
  ├─ [9] PackGpuNodes() → LBVHNodeGpu[]    [Burst 并行，叶/内部节点打包]
  │
  └─ [10] UpdateBuffers()
           ComputeBuffer "_BVH_NodeEdge_Buffer"  ← LBVHNodeGpu[]
           ComputeBuffer "_BVH_Material_Buffer"  ← MaterialData[]
           int           "_BVH_Root_Index"       ← BVH 根节点索引
           Texture2D     "_RCWB_Atlas"            ← Atlas 纹理
```

---

## 4. 关键数据结构

### `edgeBVH`（BVH 叶节点原始数据）
```csharp
struct edgeBVH {
    Vector2 start;   // 世界坐标
    Vector2 end;     // 世界坐标
    int     matIdx;  // 对应 spriteRenderers/rcwObjects 列表的索引
}
```

### `LBVHNodeGpu`（上传 GPU 的紧凑格式，16+16 字节）
```
内部节点：PosA=AABBMin, PosB=AABBMax, IndexData=LeftChild(>=0), RightChild=RightChild
叶子节点：PosA=edge.start, PosB=edge.end, IndexData=~matIdx(<0), RightChild=0（未用）
```
> Shader 中用 `IndexData < 0` 判断叶子节点，`~IndexData` 还原材质索引。

### `MaterialData`（GPU 材质，每物体一条）
```csharp
struct MaterialData {
    Color   BasicColor;      // 基础色
    Color   Emission;        // 自发光（HDR）
    Vector4 uvMatrix;        // 世界→Atlas UV 的 2×2 线性变换 (m00,m01,m10,m11)
    Vector2 uvTranslation;   // UV 平移
    float   Density;         // 物质密度
    int     TextureIndex;    // 图集索引（当前固定为 0）
    // 8 字节 padding
}
```

### `RCWBContourProfile` / `ContourLoopData`
```csharp
// ScriptableObject，存储若干轮廓环
ContourLoopData {
    bool             closed;        // 闭合环 or 折线
    List<Vector2>    pointsLocal;   // 局部 Unity 单位空间（见坐标空间说明）
}
```

---

## 5. 坐标空间（重要）

| 坐标空间 | 定义 | 用于 |
|----------|------|------|
| **源像素空间** | 以 Sprite 在源纹理中左下角为原点，单位为像素 | 轮廓追踪算法的格点坐标 |
| **局部 Unity 单位空间** | `(pixelPos − sprite.pivot) / PPU`，以 Pivot 为原点，单位为 Unity 单位 | `ContourLoopData.pointsLocal`；与 `Sprite.GetPhysicsShape()` 输出一致 |
| **世界空间** | `transform.TransformPoint(localUnits)` | `edgeBVH.start/end`；BVH 构建输入 |
| **Atlas UV 空间** | `uvMatrix × worldPos + uvTranslation` | Shader 纹理采样 |

**关键公式（Editor 工具生成 Profile 时）：**
```
pointsLocal = (gridPoint - sprite.pivot) / sprite.pixelsPerUnit
```
`sprite.pivot` 与 `gridPoint` 处于同一坐标系（均以 Sprite 源 rect 左下角为原点）。

**使用时（BVH / Gizmos）：**
```csharp
Vector3 world = transform.TransformPoint(pts[i].x, pts[i].y, 0f);
// transform 的 scale/rotation/translation 已全部包含，无需额外 PPU 处理
```

---

## 6. RCWBObject 组件字段

```csharp
bool              IsWall           // false → 不参与 BVH，不注册
RCWBContourProfile ContourProfile  // 自定义轮廓（留空则 fallback）
Color             BasicColor
float             Density
Color             Emission          // HDR
float             giCoefficient
```

**生命周期：**
- `OnEnable`：若 `IsWall=true` → 调用 `PolygonManagerCore.EnsureInitialized()` + `Register()`
- `OnDisable`：调用 `Unregister()`

**Gizmos：**
- 未选中：半透明绿色（alpha=0.4）
- 选中：不透明亮绿（alpha=1.0）
- 优先绘制 `ContourProfile`，无效时绘制 Sprite 物理形状

---

## 7. PolygonManagerSettings（Resources/PolygonManagerSettings.asset）

```csharp
Texture2D atlasTexture         // 固定 Atlas 纹理（推荐）；留空则取第一个 Sprite 的 texture
Vector4   sceneAABB            // 莫顿码归一化范围，默认 (-100,-100,100,100)
string    defaultProfileFolder // ContourProfile 输出根目录（留空 = 方案A，填路径 = 方案D）
bool      enableDebugLog
```

---

## 8. ContourProfile 资产路径规则

| 条件 | 路径 |
|------|------|
| `defaultProfileFolder` 为空（方案A） | `{SceneDir}/{SceneName}_RCWBProfiles/{GameObjectName}_ContourProfile.asset` |
| `defaultProfileFolder` 已填写（方案D） | `{defaultProfileFolder}/{SceneName}/{GameObjectName}_ContourProfile.asset` |

场景未保存时无法生成（路径不确定），工具会给出警告。

---

## 9. 编辑器工具

### `Tools/RCWB/Contour Profile Generator`
- 选中场景中的 `RCWBObject` 自动填入目标字段
- 从 Sprite **源纹理**（非 Atlas）读取像素 → Alpha 阈值二值化 → 轮廓追踪 → 去共线简化 → 转换到局部 Unity 单位空间 → 写入 `RCWBContourProfile` 并赋值
- 批量按钮：处理场景中所有 `IsWall=true && sprite!=null` 的 RCWBObject，静默模式（失败记日志不弹窗，已有文件自动覆盖）

### `Tools/RCWB/Legacy/Sprite Physics Shape Generator`
- 对单张 Sprite 生成物理形状（写回 Sprite 资产，非 Profile）
- 坐标系：以 Sprite Rect **中心**为原点，单位为像素（`ISpritePhysicsOutlineDataProvider` 的要求）
- 批量按钮：对场景中所有合法 RCWBObject 的**去重** Sprite 生成物理形状

---

## 10. Shader 全局变量

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `_RCWB_Atlas` | `Texture2D` | 合并 Atlas 纹理 |
| `_BVH_NodeEdge_Buffer` | `StructuredBuffer<LBVHNodeGpu>` | BVH 节点（内部+叶子） |
| `_BVH_Material_Buffer` | `StructuredBuffer<MaterialData>` | 材质数据 |
| `_BVH_Root_Index` | `int` | BVH 根节点在 Buffer 中的索引（BFS 排列后固定为 0） |

---

## 11. 调试

```csharp
// 运行时输出每个物体的轮廓来源
PolygonManagerCore.Instance.LogContourSourceInfo();

// 或在 Inspector 右键 PolygonManager 组件（已废弃组件，仅用于 Gizmos 可视化）
// Context Menu → "Log Contour Source Info"
```

输出格式：
```
[RCWB BVH] [0] Rock_A → 自定义 ContourProfile ("Rock_A_ContourProfile", 1 个环, 12 条边)
[RCWB BVH] [1] Rock_B → Sprite 物理形状 (fallback) (1 个轮廓, 原因: ContourProfile 未赋值)
```

---

## 12. 注意事项 / 已知约束

1. **`PolygonManager`（旧版）已废弃**：不要挂载到场景，会覆盖 `PolygonManagerCore` 上传的 GPU 数据。
2. **`rcwObjects` 与 `spriteRenderers` 严格同序**：`matIdx` 用于索引 `MaterialData`，两表必须保持一致，禁止单独修改其中一个。
3. **同一 Sprite 不同 Profile**：`ContourProfile` 存储在 `RCWBObject`（场景实例）而非 Sprite 资产上，多个物体共用同一 Sprite 可以有各自独立的轮廓。
4. **Atlas 绑定顺序敏感**：推荐在 `PolygonManagerSettings.atlasTexture` 中显式指定 Atlas，避免依赖 `spriteRenderers[0]` 的顺序。
5. **BVH 每帧重建**：当前设计是每帧完整重建 BVH（适合动态场景）；静态场景可考虑加脏标记跳过不变帧。
