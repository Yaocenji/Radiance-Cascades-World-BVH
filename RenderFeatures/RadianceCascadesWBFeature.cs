using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace RadianceCascadesWorldBVH
{

    [Serializable]
    public class RcwbSettings
    {
        [Header("基础渲染设置")]
        public float renderScale;
        public int cascadeCount;
        public float rayRange;
        public float bounceIntensity;

        [Header("天空与太阳")]
        public Color skyColor;
        public float skyIntensity;
        public Color sunColor;
        public float sunAngle;
        public float sunIntensity;
        public float sunHardness;

        [Header("模糊设置")]
        public int blurIterations = 4;      // 控制降采样的深度，4 是个非常好的默认值
        public float blurRadius = 1.0f;     // 散布半径

        [Header("全局光高度")]
        public float giHeight = 1.0f;

        [Header("Shading Options")]
        [Tooltip("Enable translucent-object branch in shader space (keyword: ENABLE_TRANSLUCENT_OBJECTS).")]
        public bool enableTranslucentObjects = false;

        public RcwbSettings(
            float renderScale,
            int cascadeCount,
            float rayRange,
            float bounceIntensity,
            Color skyColor,
            float skyIntensity,
            Color sunColor,
            float sunAngle,
            float sunIntensity,
            float sunHardness,
            float giHeight)
        {
            this.renderScale = renderScale;
            this.cascadeCount = cascadeCount;
            this.rayRange = rayRange;
            this.bounceIntensity = bounceIntensity;
            this.skyColor = skyColor;
            this.skyIntensity = skyIntensity;
            this.sunColor = sunColor;
            this.sunAngle = sunAngle;
            this.sunIntensity = sunIntensity;
            this.sunHardness = sunHardness;
            this.giHeight = giHeight;
        }
    }
    
    public class RadianceCascadesWBFeature : ScriptableRendererFeature
    {
        private const string TranslucentObjectsKeyword = "ENABLE_TRANSLUCENT_OBJECTS";
        public ComputeShader rcShader;
        public RcwbSettings settings;

        // Feature 级共享字段：RcwbRenderPass 写入本帧结果，SceneColorHistoryPass 在更晚阶段读取
        internal RTHandle FrameLightResult;
        internal RTHandle FrameLightResultBlur;
        
        class RcwbRenderPass : ScriptableRenderPass
        {
            public ComputeShader rcShader;
            public RcwbSettings settings;

            // Feature 引用，用于写入本帧共享信息
            private readonly RadianceCascadesWBFeature m_Feature;

            // 摄像机的引用
            private Camera m_Camera;

            // 跨帧持久的 history RT（存储上一帧 RC light result）
            private RTHandle m_HistoryRT_Blur;
            private RTHandle m_HistoryRT_NoBlur;
            public float historyScale = 1.0f;
            public float historyWeight = 0.85f;
            
            // RT的引用
            private RTHandle m_Rcwb_Handle_0;
            private RTHandle m_Rcwb_Handle_1;
            private RTHandle m_Rcwb_TemporalHandle_0;
            private RTHandle m_Rcwb_TemporalHandle_1;
            //private RTHandle m_Rcwb_Handle_StencilFullResolution;
            private RTHandle m_Rcwb_Direction;
            private RTHandle m_Rcwb_Direction_Blur;
            private RTHandle[] m_KawasePyramid;
            
            // 上一帧的 VP / InvVP 矩阵（用于 reprojection）
            private Matrix4x4 m_PrevViewProjMatrix    = Matrix4x4.identity;
            private Matrix4x4 m_PrevViewProjMatrixInv = Matrix4x4.identity;

            // 当前的目标分辨率与RC分辨率
            private int fullWidth;
            private int fullHeight;
            private int width;
            private int height;
            private int rcWidth;
            private int rcHeight;

            public RcwbRenderPass(RcwbSettings settings, ComputeShader rcShader, RadianceCascadesWBFeature feature)
            {
                this.settings = settings;
                this.rcShader = rcShader;
                m_Feature = feature;
            }
            
            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
                m_Camera = renderingData.cameraData.camera;
                var cameraTargetDescriptor = renderingData.cameraData.cameraTargetDescriptor;
                
                fullWidth = cameraTargetDescriptor.width;
                fullHeight = cameraTargetDescriptor.height;

                // 计算目标分辨率（支持动态画质调整）
                width = (int)(cameraTargetDescriptor.width * settings.renderScale);
                height = (int)(cameraTargetDescriptor.height * settings.renderScale);
                
                // 计算rc分辨率
                int blockSize = (int)Mathf.Pow(2, settings.cascadeCount);
                rcWidth = Mathf.CeilToInt((float)width / blockSize) * blockSize;
                rcHeight = Mathf.CeilToInt((float)height / blockSize) * blockSize;
                
                // 配置 Radiance Buffer 描述符
                var radianceDesc = new RenderTextureDescriptor(rcWidth, rcHeight, RenderTextureFormat.ARGBHalf, 0);
                radianceDesc.msaaSamples = 1;
                radianceDesc.sRGB = false;
                radianceDesc.useMipMap = false;
                radianceDesc.enableRandomWrite = true;

                // 配置 Radiance Full Resolution 描述符
                var radianceFullResolutionDesc = new RenderTextureDescriptor(fullWidth, fullHeight, RenderTextureFormat.RHalf, 0);
                radianceFullResolutionDesc.msaaSamples = 1;
                radianceFullResolutionDesc.sRGB = false;
                radianceFullResolutionDesc.useMipMap = false;
                radianceFullResolutionDesc.enableRandomWrite = true;

                // 配置 Radiance Direction 描述符
                var radianceDirectionDesc = new RenderTextureDescriptor(rcWidth, rcHeight, RenderTextureFormat.RGHalf, 0);
                radianceDirectionDesc.msaaSamples = 1;
                radianceDirectionDesc.sRGB = false;
                radianceDirectionDesc.useMipMap = false;
                radianceDirectionDesc.enableRandomWrite = true;
                
                // 动态地分配显存
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_0, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_0");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_1, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_1");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_TemporalHandle_0, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_Temporal_0");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_TemporalHandle_1, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_Temporal_1");
                //RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_StencilFullResolution, radianceFullResolutionDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_FullResolution");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Direction, radianceDirectionDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_Direction");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Direction_Blur, radianceDirectionDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_Direction_Blur");

                // 历史帧 RT：存储上一帧 RC light result，用于多次弹射
                int histW = Mathf.Max(1, Mathf.RoundToInt(rcWidth * historyScale));
                int histH = Mathf.Max(1, Mathf.RoundToInt(rcHeight * historyScale));
                var historyDesc = new RenderTextureDescriptor(histW, histH, RenderTextureFormat.ARGBHalf, 0)
                {
                    msaaSamples = 1,
                    sRGB = false,
                    useMipMap = false
                };
                RenderingUtils.ReAllocateIfNeeded(ref m_HistoryRT_Blur, historyDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_HistoryLight_Blur");
                
                RenderingUtils.ReAllocateIfNeeded(ref m_HistoryRT_NoBlur, historyDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_HistoryLight_NoBlur");
                

                // 特殊的：分配降采样纹理的金字塔缓存
                int maxIterations = settings.blurIterations;
                if (m_KawasePyramid == null || m_KawasePyramid.Length != maxIterations)
                {
                    m_KawasePyramid = new RTHandle[maxIterations];
                }
                int mipWidth = rcWidth / 2;
                int mipHeight = rcHeight / 2;
                var mipDesc = radianceDesc; // 继承之前配置好的 Descriptor
                for (int i = 0; i < maxIterations; i++)
                {
                    mipDesc.width = Mathf.Max(1, mipWidth);
                    mipDesc.height = Mathf.Max(1, mipHeight);
                    
                    RenderingUtils.ReAllocateIfNeeded(
                        ref m_KawasePyramid[i], 
                        mipDesc, 
                        FilterMode.Bilinear, 
                        TextureWrapMode.Clamp, 
                        name: $"_RCWB_KawaseMip_{i}"
                    );

                    mipWidth /= 2;
                    mipHeight /= 2;
                }
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                // 写个名字
                CommandBuffer cmd = CommandBufferPool.Get("Radiance Cascades");
                if (settings != null && settings.enableTranslucentObjects)
                {
                    cmd.EnableShaderKeyword(TranslucentObjectsKeyword);
                }
                else
                {
                    cmd.DisableShaderKeyword(TranslucentObjectsKeyword);
                }
                
                cmd.BeginSample("Radiance Cascades WB");
                
                int rcMainKernelHandle = rcShader.FindKernel("RadianceCascadeMain");
                
                // 摄像机的VP矩阵及其逆矩阵
                Matrix4x4 view = m_Camera.worldToCameraMatrix;
                Matrix4x4 proj = GL.GetGPUProjectionMatrix(m_Camera.projectionMatrix, true);
                Matrix4x4 viewProjMatrix = proj * view;
                Matrix4x4 viewProjMatrixInv = Matrix4x4.Inverse(viewProjMatrix);
                cmd.SetComputeMatrixParam(rcShader, "MatrixVP", viewProjMatrix);
                cmd.SetComputeMatrixParam(rcShader, "MatrixInvVP", viewProjMatrixInv);
                cmd.SetComputeMatrixParam(rcShader, "MatrixVP_Prev", m_PrevViewProjMatrix);

                // 显式绑定历史帧纹理到 compute kernel
                if (m_HistoryRT_Blur != null)
                {
                    cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "_RCWB_HistoryColor", m_HistoryRT_Blur);
                    int histW = m_HistoryRT_Blur.rt.width;
                    int histH = m_HistoryRT_Blur.rt.height;
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_HistoryColor_TexelSize", new Vector4(1f / histW, 1f / histH, histW, histH));
                }
                
                if (m_HistoryRT_NoBlur != null)
                {
                    cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "_RCWB_HistoryColor_NoBlur", m_HistoryRT_NoBlur);
                    int histW = m_HistoryRT_NoBlur.rt.width;
                    int histH = m_HistoryRT_NoBlur.rt.height;
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_HistoryColor_NoBlur_TexelSize", new Vector4(1f / histW, 1f / histH, histW, histH));
                }
                
                cmd.SetComputeVectorParam(rcShader, "_CameraResolution_Full", new Vector2(fullWidth, fullHeight));
                cmd.SetComputeVectorParam(rcShader, "_CameraResolution_Resized", new Vector2(width, height));

                // RC分辨率
                cmd.SetComputeVectorParam(rcShader, "_RCWB_CascadeResolution", new Vector2(rcWidth, rcHeight));
                
                // RC最大层级
                cmd.SetComputeIntParam(rcShader, "_RCWB_CascadeCount", settings.cascadeCount);
                
                // 射线范围
                cmd.SetComputeFloatParam(rcShader, "_RCWB_RayRange_WS", settings.rayRange);
                cmd.SetComputeVectorParam(rcShader, "_RCWB_SkyColor", settings.skyColor);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SkyIntensity", settings.skyIntensity);
                cmd.SetComputeVectorParam(rcShader, "_RCWB_SunColor", settings.sunColor);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SunAngle", settings.sunAngle);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SunIntensity", settings.sunIntensity);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SunHardness", settings.sunHardness);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_BounceIntensity", settings.bounceIntensity);

                for (int i = 0; i < settings.cascadeCount; i++)
                {
                    // RC的当前层级
                    cmd.SetComputeIntParam(rcShader, "_RCWB_CascadeLevel", settings.cascadeCount - i - 1);
                    // RC的 merge 纹理
                    cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "_RCWB_LastResult", (i & 1) == 1 ?  m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                    // RC的目标纹理
                    cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "Result", (i & 1) == 1 ?  m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                    cmd.DispatchCompute(rcShader, rcMainKernelHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);
                }
                
                cmd.EndSample("Radiance Cascades WB");
                cmd.BeginSample("Calculate Direction");

                int directionKernalHandle = rcShader.FindKernel("DirectionMain");
                cmd.SetComputeTextureParam(rcShader, directionKernalHandle, "Direction", m_Rcwb_Direction);
                cmd.SetComputeTextureParam(rcShader, directionKernalHandle, "_RCWB_OneDirResult", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                cmd.SetComputeTextureParam(rcShader, directionKernalHandle, "_RCWB_FourDirResult", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                cmd.DispatchCompute(rcShader, directionKernalHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.EndSample("Calculate Direction");
                // Dual Kawase Blur 阶段
                cmd.BeginSample("Invasion");

                int invasionLightKernalHandle = rcShader.FindKernel("InvasionLightMain");
                cmd.SetComputeTextureParam(rcShader, invasionLightKernalHandle, "_RCWB_LightResult_NonInvasion", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                cmd.SetComputeTextureParam(rcShader, invasionLightKernalHandle, "_RCWB_LightResult_Invasion", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                cmd.DispatchCompute(rcShader, invasionLightKernalHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.CopyTexture(((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0, 
                                ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1);

                int invasionDirectionKernalHandle = rcShader.FindKernel("InvasionDirectionMain");
                cmd.SetComputeTextureParam(rcShader, invasionDirectionKernalHandle, "_RCWB_Direction_NonInvasion", m_Rcwb_Direction);
                cmd.SetComputeTextureParam(rcShader, invasionDirectionKernalHandle, "_RCWB_Direction_Invasion", m_Rcwb_Direction_Blur);
                cmd.DispatchCompute(rcShader, invasionDirectionKernalHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.CopyTexture(m_Rcwb_Direction_Blur, m_Rcwb_Direction);
                
                cmd.EndSample("Invasion");
                cmd.BeginSample("Dual Kawase Blur");

                ExecuteDualKawaseBlur(cmd, ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1, ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0, rcWidth, rcHeight, settings.blurIterations, settings.blurRadius);
                ExecuteDualKawaseBlur(cmd, m_Rcwb_Direction, m_Rcwb_Direction_Blur, rcWidth, rcHeight, settings.blurIterations, settings.blurRadius);
                cmd.EndSample("Dual Kawase Blur");
                cmd.BeginSample("Temporal Accumulation");

                var currentLightResultHandle = ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1;
                var currentLightResultHandleBlur = ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0;
                var accumulatedLightResultHandle = m_Rcwb_TemporalHandle_0;
                var accumulatedLightResultHandleBlur = m_Rcwb_TemporalHandle_1;
                int temporalAccumulateKernelHandle = rcShader.FindKernel("TemporalAccumulateMain");
                cmd.SetComputeTextureParam(rcShader, temporalAccumulateKernelHandle, "_RCWB_TemporalCurrentNoBlur", currentLightResultHandle);
                cmd.SetComputeTextureParam(rcShader, temporalAccumulateKernelHandle, "_RCWB_TemporalCurrentBlur", currentLightResultHandleBlur);
                if (m_HistoryRT_Blur != null)
                {
                    cmd.SetComputeTextureParam(rcShader, temporalAccumulateKernelHandle, "_RCWB_HistoryColor", m_HistoryRT_Blur);
                    int histW = m_HistoryRT_Blur.rt.width;
                    int histH = m_HistoryRT_Blur.rt.height;
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_HistoryColor_TexelSize", new Vector4(1f / histW, 1f / histH, histW, histH));
                }
                if (m_HistoryRT_NoBlur != null)
                {
                    cmd.SetComputeTextureParam(rcShader, temporalAccumulateKernelHandle, "_RCWB_HistoryColor_NoBlur", m_HistoryRT_NoBlur);
                    int histW = m_HistoryRT_NoBlur.rt.width;
                    int histH = m_HistoryRT_NoBlur.rt.height;
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_HistoryColor_NoBlur_TexelSize", new Vector4(1f / histW, 1f / histH, histW, histH));
                }
                cmd.SetComputeFloatParam(rcShader, "_RCWB_HistoryWeight", historyWeight);
                cmd.SetComputeTextureParam(rcShader, temporalAccumulateKernelHandle, "_RCWB_TemporalAccumulatedNoBlur", accumulatedLightResultHandle);
                cmd.SetComputeTextureParam(rcShader, temporalAccumulateKernelHandle, "_RCWB_TemporalAccumulatedBlur", accumulatedLightResultHandleBlur);
                cmd.DispatchCompute(rcShader, temporalAccumulateKernelHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.EndSample("Temporal Accumulation");
                cmd.BeginSample("After RCWB");

                // 获取颜色结果
                var lightResultHandle = accumulatedLightResultHandle;
                var lightResultHandleBlur = accumulatedLightResultHandleBlur;
                
                //RTHandle cameraColorTargetHandle = renderingData.cameraData.renderer.cameraColorTargetHandle;
                //Blitter.BlitCameraTexture(cmd, m_Rcwb_Direction, cameraColorTargetHandle, 0f, true);

                // 将RC变量绑定到全局，让后续能访问到
                cmd.SetGlobalMatrix("MatrixVP", viewProjMatrix);
                cmd.SetGlobalMatrix("MatrixInvVP", viewProjMatrixInv);
                cmd.SetGlobalMatrix("MatrixVP_Prev", m_PrevViewProjMatrix);
                cmd.SetGlobalMatrix("MatrixInvVP_Prev", m_PrevViewProjMatrixInv);
                // 保存本帧矩阵供下一帧使用
                m_PrevViewProjMatrix    = viewProjMatrix;
                m_PrevViewProjMatrixInv = viewProjMatrixInv;
                cmd.SetGlobalVector("_CameraResolution_Full", new Vector2(fullWidth, fullHeight));
                cmd.SetGlobalVector("_CameraResolution_Resized", new Vector2(width, height));
                cmd.SetGlobalVector("_RCWB_CascadeResolution", new Vector2(rcWidth, rcHeight));
                cmd.SetGlobalInt("_RCWB_CascadeCount", settings.cascadeCount);
                cmd.SetGlobalFloat("_RCWB_RayRange_WS", settings.rayRange);
                // 包含全局光的高度
                cmd.SetGlobalFloat("_RCWB_GI_Height", settings.giHeight);

                // 最后的最后：将光源和方向的纹理绑定，让后续的渲染中能够采样到
                cmd.SetGlobalTexture("_RCWB_LightResult", lightResultHandle);
                cmd.SetGlobalTexture("_RCWB_LightResult_Blur", lightResultHandleBlur);
                cmd.SetGlobalTexture("_RCWB_DirectionResult", m_Rcwb_Direction);
                cmd.SetGlobalTexture("_RCWB_DirectionResult_Blur", m_Rcwb_Direction_Blur);

                // 将本帧结果写入 Feature 共享字段，供 SceneColorHistoryPass 在 AfterRenderingTransparents 时写入 history
                m_Feature.FrameLightResult     = lightResultHandle;
                m_Feature.FrameLightResultBlur = lightResultHandleBlur;

                // history RT 仍在此处绑定到全局（内容是上一帧，供本帧 RC kernel 采样用）
                if (m_HistoryRT_Blur != null)
                {
                    cmd.SetGlobalTexture("_RCWB_HistoryColor", m_HistoryRT_Blur);
                    int hw = m_HistoryRT_Blur.rt.width;
                    int hh = m_HistoryRT_Blur.rt.height;
                    cmd.SetGlobalVector("_RCWB_HistoryColor_TexelSize", new Vector4(1f / hw, 1f / hh, hw, hh));
                }
                if (m_HistoryRT_NoBlur != null)
                {
                    cmd.SetGlobalTexture("_RCWB_HistoryColor_NoBlur", m_HistoryRT_NoBlur);
                    int hw = m_HistoryRT_NoBlur.rt.width;
                    int hh = m_HistoryRT_NoBlur.rt.height;
                    cmd.SetGlobalVector("_RCWB_HistoryColor_NoBlur_TexelSize", new Vector4(1f / hw, 1f / hh, hw, hh));
                }

                cmd.SetGlobalFloat("_RCWB_HistoryWeight", historyWeight);
                

                cmd.EndSample("After RCWB");
                
                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
                
            }

            private void ExecuteDualKawaseBlur(CommandBuffer cmd, RTHandle srcHandle, RTHandle dstHandle, int width, int height, int iterations, float radius)
            {
                // 1. 健壮性检查：如果金字塔未分配，或者要求迭代次数 <= 0
                if (m_KawasePyramid == null || m_KawasePyramid.Length == 0 || iterations <= 0)
                {
                    // 直接将输入拷贝到输出（前提是它们不是同一个RT）
                    if (srcHandle != dstHandle)
                    {
                        Blitter.BlitCameraTexture(cmd, srcHandle, dstHandle, 0f, true);
                    }
                    return;
                }

                // 2. 防止传入的迭代次数超过实际分配的金字塔层数
                int actualIterations = Mathf.Min(iterations, m_KawasePyramid.Length);

                int downKernel = rcShader.FindKernel("KawaseDownsample");
                int upKernel = rcShader.FindKernel("KawaseUpsample");

                RTHandle currentSrc = srcHandle;
                int currentSrcWidth = width;
                int currentSrcHeight = height;

                // ==========================================
                // 阶段 1: 降采样 (Downsample)
                // ==========================================
                for (int i = 0; i < actualIterations; i++)
                {
                    RTHandle currentDst = m_KawasePyramid[i];
                    int dstWidth = currentDst.rt.width;
                    int dstHeight = currentDst.rt.height;

                    cmd.SetComputeVectorParam(rcShader, "_RCWB_InputResolution", new Vector2(currentSrcWidth, currentSrcHeight));
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_OutputResolution", new Vector2(dstWidth, dstHeight));
                    cmd.SetComputeTextureParam(rcShader, downKernel, "_RCWB_BlurInput", currentSrc);
                    cmd.SetComputeTextureParam(rcShader, downKernel, "_RCWB_BlurOutput", currentDst);
                    
                    cmd.DispatchCompute(rcShader, downKernel, (dstWidth + 7) / 8, (dstHeight + 7) / 8, 1);

                    currentSrc = currentDst;
                    currentSrcWidth = dstWidth;
                    currentSrcHeight = dstHeight;
                }

                // 传入自定义的模糊散布半径
                cmd.SetComputeFloatParam(rcShader, "_RCWB_BlurRadius", radius);

                // ==========================================
                // 阶段 2: 升采样 (Upsample) - 直到倒数第二次
                // ==========================================
                for (int i = actualIterations - 2; i >= 0; i--)
                {
                    RTHandle currentDst = m_KawasePyramid[i];
                    int dstWidth = currentDst.rt.width;
                    int dstHeight = currentDst.rt.height;

                    cmd.SetComputeVectorParam(rcShader, "_RCWB_InputResolution", new Vector2(currentSrcWidth, currentSrcHeight));
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_OutputResolution", new Vector2(dstWidth, dstHeight));
                    cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurInput", currentSrc);
                    cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurOutput", currentDst);
                    
                    cmd.DispatchCompute(rcShader, upKernel, (dstWidth + 7) / 8, (dstHeight + 7) / 8, 1);

                    currentSrc = currentDst;
                    currentSrcWidth = dstWidth;
                    currentSrcHeight = dstHeight;
                }

                // ==========================================
                // 阶段 3: 最后一次升采样 (直接输出到 dstHandle)
                // ==========================================
                cmd.SetComputeVectorParam(rcShader, "_RCWB_InputResolution", new Vector2(currentSrcWidth, currentSrcHeight));
                cmd.SetComputeVectorParam(rcShader, "_RCWB_OutputResolution", new Vector2(width, height));
                cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurInput", currentSrc);
                cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurOutput", dstHandle);
                
                cmd.DispatchCompute(rcShader, upKernel, (width + 7) / 8, (height + 7) / 8, 1);
            }

            public RTHandle HistoryRTBlur => m_HistoryRT_Blur;
            public RTHandle HistoryRTNoBlur => m_HistoryRT_NoBlur;

            public override void OnCameraCleanup(CommandBuffer cmd)
            {
            }

            public void Dispose()
            {
                m_HistoryRT_Blur?.Release();
                m_HistoryRT_Blur = null;
                m_HistoryRT_NoBlur?.Release();
                m_HistoryRT_NoBlur = null;
                m_Rcwb_TemporalHandle_0?.Release();
                m_Rcwb_TemporalHandle_0 = null;
                m_Rcwb_TemporalHandle_1?.Release();
                m_Rcwb_TemporalHandle_1 = null;
            }
        }

        // =====================================================================
        // Pass：在 Transparent 之后将本帧 RC 结果写入 history RT
        // =====================================================================
        class SceneColorHistoryPass : ScriptableRenderPass
        {
            private readonly RadianceCascadesWBFeature m_Feature;
            private readonly RcwbRenderPass m_RcwbRenderPass;

            public SceneColorHistoryPass(RadianceCascadesWBFeature feature, RcwbRenderPass rcwbRenderPass)
            {
                m_Feature = feature;
                m_RcwbRenderPass = rcwbRenderPass;
                renderPassEvent = RenderPassEvent.AfterRenderingTransparents;
            }

            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                if (m_Feature.FrameLightResult == null || m_Feature.FrameLightResultBlur == null)
                {
                    return;
                }

                CommandBuffer cmd = CommandBufferPool.Get("Scene Color History");

                if (m_RcwbRenderPass.HistoryRTBlur != null)
                {
                    Blitter.BlitCameraTexture(cmd, m_Feature.FrameLightResultBlur, m_RcwbRenderPass.HistoryRTBlur);
                    cmd.SetGlobalTexture("_RCWB_HistoryColor", m_RcwbRenderPass.HistoryRTBlur);
                    int w = m_RcwbRenderPass.HistoryRTBlur.rt.width;
                    int h = m_RcwbRenderPass.HistoryRTBlur.rt.height;
                    cmd.SetGlobalVector("_RCWB_HistoryColor_TexelSize", new Vector4(1f / w, 1f / h, w, h));
                }

                if (m_RcwbRenderPass.HistoryRTNoBlur != null)
                {
                    Blitter.BlitCameraTexture(cmd, m_Feature.FrameLightResult, m_RcwbRenderPass.HistoryRTNoBlur);
                    cmd.SetGlobalTexture("_RCWB_HistoryColor_NoBlur", m_RcwbRenderPass.HistoryRTNoBlur);
                    int w = m_RcwbRenderPass.HistoryRTNoBlur.rt.width;
                    int h = m_RcwbRenderPass.HistoryRTNoBlur.rt.height;
                    cmd.SetGlobalVector("_RCWB_HistoryColor_NoBlur_TexelSize", new Vector4(1f / w, 1f / h, w, h));
                }

                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
            }

            public override void OnCameraCleanup(CommandBuffer cmd) { }

            public void Dispose()
            {
            }
        }

        // =====================================================================

        RcwbRenderPass m_ScriptablePass;
        SceneColorHistoryPass m_SceneColorHistoryPass;

        [Header("历史帧 (多次弹射)")]
        [Tooltip("历史帧 RT 相对 RC 分辨率的缩放系数。1.0 = 与 RC 同分辨率")]
        [Range(0.25f, 1f)]
        public float historyScale = 1.0f;

        [Tooltip("历史帧 的 比重")] [Range(0.0f, 0.99f)]
        public float historyWeight = .85f;

        /// <inheritdoc/>
        public override void Create()
        {
            m_ScriptablePass = new RcwbRenderPass(settings, rcShader, this);
            m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingOpaques;

            m_SceneColorHistoryPass = new SceneColorHistoryPass(this, m_ScriptablePass);

            ApplyShaderKeywords();
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            // 判定当前渲染窗口是哪个
            ApplyShaderKeywords();
            var cameraType = renderingData.cameraData.cameraType;
            if (cameraType == CameraType.Preview || cameraType == CameraType.Reflection || cameraType == CameraType.SceneView)
            {
                return;
            }

            // 每帧同步可能在 Inspector 中调整的参数
            m_ScriptablePass.historyScale = historyScale;
            m_ScriptablePass.historyWeight = historyWeight;

            // 只有 game 窗口会应用 renderPass
            renderer.EnqueuePass(m_ScriptablePass);
            renderer.EnqueuePass(m_SceneColorHistoryPass);
        }

        protected override void Dispose(bool disposing)
        {
            m_ScriptablePass?.Dispose();
            m_SceneColorHistoryPass?.Dispose();
        }

        private void ApplyShaderKeywords()
        {
            if (settings != null && settings.enableTranslucentObjects)
            {
                Shader.EnableKeyword(TranslucentObjectsKeyword);
            }
            else
            {
                Shader.DisableKeyword(TranslucentObjectsKeyword);
            }
        }
    }
        
}


