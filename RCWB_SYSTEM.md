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
│   ├── PolygonManager.cs                 # [OBSOLETE] 旧 MonoBehaviour 管理器，已废弃
│   └── Boolean/
│       ├── Mesh.cs                       # 布尔算法数据结构（Vertex/Edge/Loop/Mesh）
│       └── BooleanOperation.cs           # 布尔算法核心（Intersection / Addition）
├── Editor/
│   ├── RCWBContourProfileGenerator.cs    # 编辑器工具：从 Sprite 像素生成 ContourProfile
│   ├── SpritePhysicsShapeGenerator.cs    # 编辑器工具（Legacy）：生成 Sprite 物理形状
│   ├── RCWBContourProfileEditor.cs       # ContourProfile 的自定义 Inspector
│   └── RCWBBooleanSubtractTool.cs        # 编辑器工具：多对多布尔减法，结果写回 ContourProfile
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

### `Tools/RCWB/Boolean Subtract (A - B)`
- 对两组 RCWBObject 执行**多对多布尔减法**：A 组每个物体的轮廓，依次减去 B 组所有物体的轮廓，结果写回 A 物体的 `ContourProfile`；B 组物体不受影响
- 详见第 13 节

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

---

## 13. 布尔减法编辑器工具（RCWBBooleanSubtractTool）

### 13.1 用途

当两个 RCWBObject 的轮廓在世界空间中存在重叠时，BVH 中会出现完全共线的边，导致光照计算精度问题。布尔减法工具用于**预处理轮廓**：让 A 的轮廓减去 B 的轮廓，使两者在边界处分开一条极细缝，彻底消除重叠。

### 13.2 操作语义

**A 组中每个物体**的轮廓，依次减去 **B 组中所有物体**的轮廓，结果写回该 A 物体的 `ContourProfile`。B 组物体不受影响。

数学表达（对单个 A_i）：

```
result = A_i ∩ (¬inflated_B_1) ∩ (¬inflated_B_2) ∩ … ∩ (¬inflated_B_n)
```

即链式相减，每次用上一步结果作为新的 A 输入。

### 13.3 等距细缝：先膨胀 B，再做减法

直接做减法会使 A 的新边界与 B 的原始边界**精确重合**，仍有精度问题。  
解决方案：在布尔运算之前，先将 B 的轮廓向外膨胀 `Epsilon`，使 A−B 的结果与原始 B 之间自然保留等宽间隙。

**膨胀算法（Miter Join）：**

每个顶点沿**顶点外法线（两侧边外法线的角平分线）**移动距离 d：

```
d = ε / sin(α/2)
```

其中 α 为该顶点的多边形内角，ε 为目标垂直间距（`Epsilon` 参数）。  
此公式保证膨胀后对两侧边的**垂直间距均等于 ε**（等距细缝）。

| 内角 α | d（相对 ε） | 说明 |
|--------|------------|------|
| 180°（平直） | 1.0ε | 无尖角 |
| 90°（直角） | 1.41ε | 方角 |
| 60°（三角） | 2.0ε | 尖角需多走 |
| ≈ 0°（退化） | → ∞ → clamp | 由 `Miter 上限` 限制 |

**前提假设（无需运行时判断绕向）：**  
所有 Loop 均为 CCW（逆时针），由 `RCWBContourProfileGenerator` 的追踪算法规范保证。  
因此外法线方向 = 边方向的右手垂直：`edge dir (dx,dy) → outward normal (dy,−dx)`。

**退化情形处理：**  
若两侧外法线之和趋近零向量（近似折回尖刺，`|nPrev+nNext|² < 1e-6`），  
直接以前一法线方向偏移 ε，避免除零。

### 13.4 执行流程

```
Execute(groupA, groupB, epsilon, miterLimit)
  for each objA_i in groupA:
    meshCurrentA ← BuildWorldMesh(objA_i)       // ContourProfile / Sprite fallback → 世界坐标

    for each objB_j in groupB:
      meshB    ← BuildWorldMesh(objB_j)
      InflateMesh(meshB, epsilon, miterLimit)    // Miter Join 膨胀
      meshBRev ← GetReverse(meshB)              // 反转 = ¬B
      meshC    ← Intersection(meshCurrentA, meshBRev)

      if meshC 为空 AND meshCurrentA 不为空:
        testPoint ← meshCurrentA 的首个顶点
        if testPoint 在原始 meshB 外:            // IsPointInsideMesh（射线法）
          → A 与 B 无交叠，减法无效，静默跳过（保持 meshCurrentA 不变）
        else:
          → A 被 B 完全覆盖，结果正确为空，继续传播
      else:
        meshCurrentA ← meshC                    // 链式更新

    newLoops ← WorldMeshToProfileLoops(meshCurrentA, objA_i.transform)
    WriteBack(newLoops → objA_i.ContourProfile)
```

**坐标转换：**
- `BuildWorldMesh`：`pointsLocal → transform.TransformPoint() → 世界坐标 BoolMesh`
- `WorldMeshToProfileLoops`：`世界坐标 → transform.InverseTransformPoint() → pointsLocal`

**临时 GameObject 管理：**  
`BoolMesh`（`Mesh : MonoBehaviour`）不能直接 `new`，每次操作创建 `HideAndDontSave` 临时 GO 承载。所有临时 GO 记录在 `tempGOs` 列表，`finally` 块统一 `DestroyImmediate`，异常时也不泄漏。

### 13.5 无交叠检测（IsPointInsideMesh）

布尔库的 `CalculateLoopsWhenNoInoutPoints` 在 A 与 ¬B 无边相交时，会通过射线投票判断 A 是否在 ¬B 内部。由于 ¬B 是 CW 反向环，此投票可能不稳定，导致错误返回空结果。

工具在每次 `Intersection` 后检测：若结果为空，对 A 的代表点用射线法检测是否在**原始 B（未反向）**内部：

```
crossings ← 从 point 向 +X 方向射线与 B 所有边的交叉数
isInside  ← (crossings & 1) == 1
```

- **A 在 B 外**（crossings 为偶数）→ 无交叠，静默跳过，A 保持不变
- **A 在 B 内**（crossings 为奇数）→ A 被完全覆盖，结果正确为空

### 13.6 UI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Group A | （列表）| 被减物体，支持多个；每行显示轮廓来源简标 |
| Group B | （列表）| 减去物体，支持多个；B 组物体不被修改 |
| Epsilon（世界单位） | 0.005 | 目标垂直间距，典型值 0.001–0.01 |
| Miter 上限 | 4 | 最大偏移倍数 d_max/ε，限制尖角处膨胀量 |

**轮廓来源优先级（与运行时一致）：**  
`ContourProfile`（有效时）> `Sprite.GetPhysicsShape()`（fallback）

若 A 物体尚无 `ContourProfile`，工具会按 §8 路径约定自动创建并赋值，支持 Undo。
