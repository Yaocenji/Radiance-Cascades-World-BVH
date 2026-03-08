Shader "RadianceCascadesWB/RCWB_Object"
{
    Properties
    {
        [Header(Textures)]
        [PerRendererData] _MainTex ("Albedo (RGB) Alpha (A)", 2D) = "white" {}
        [PerRendererData] _BumpMap ("Normal Map", 2D) = "bump" {}

        [Header(Emission Data)]
        [HDR] _Emission ("Emission Color", Color) = (0,0,0,0)
        
        [Header(Radiance Cascades Data)]
        _IsWall ("Is Wall (1=Block Light)", Float) = 1.0
        _Occlusion ("Occlusion (0=Transparent, 1=Opaque)", Range(0.0, 1.0)) = 1.0
        
        /*[HideInInspector]
        _RC_HistoryTexture ("History Texture", 2D) = "black" {}*/
    }

    SubShader
    {
        // 渲染队列根据需要调整，通常墙壁是不透明的 (Geometry)
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        // =================================================================================
        // Pass 1: Universal2D
        // 作用：主相机渲染，玩家看到的最终画面 (Albedo + Normal Lighting)
        // =================================================================================
        Pass
        {
            Name "Universal2D"
            Tags { "Queue" = "Transparent" "LightMode"="Universal2D" }

            // 混合模式根据需求，墙壁通常是不透明 (One Zero)
            Blend SrcAlpha OneMinusSrcAlpha, One OneMinusSrcAlpha
            Cull Off
            ZWrite Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            
            // 引入 URP 核心库
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // RCWB库
            #include "RCW_BVH_Inc.hlsl"
            #include "SpotLight2D_Inc.hlsl"

            // ---------------------------------------------------------
            // 1. CBUFFER 定义 (严格匹配 SRP Batcher)
            // ---------------------------------------------------------
            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                float4 _BumpMap_ST;
                half4 _Emission;
                float2 _RotationSinCos; // x=cos, y=sin
            CBUFFER_END

            // ---------------------------------------------------------
            // 2. 纹理定义 (分离采样器以提高性能)
            // ---------------------------------------------------------
            TEXTURE2D(_MainTex);        SAMPLER(sampler_MainTex);
            TEXTURE2D(_BumpMap);        SAMPLER(sampler_BumpMap);

            // ---------------------------------------------------------
            // 3. 输入/输出 结构体
            // ---------------------------------------------------------
            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
                float3 normalOS   : NORMAL;
                float4 tangentOS  : TANGENT;
                float4 color  : COLOR;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv         : TEXCOORD0;
                float4 color : TEXCOORD1;
                half3 normalWS    : TEXCOORD2;
                half3 tangentWS   : TEXCOORD3;
                half3 bitangentWS : TEXCOORD4;
            };

            // ---------------------------------------------------------
            // 4. 顶点着色器
            // ---------------------------------------------------------
            Varyings Vert(Attributes IN)
            {
                Varyings OUT = (Varyings)0;
                // 顶点变换
                VertexPositionInputs vertexInput = GetVertexPositionInputs(IN.positionOS.xyz);
                OUT.positionCS = vertexInput.positionCS;
                OUT.uv = TRANSFORM_TEX(IN.uv, _MainTex);
                OUT.color = IN.color;

                // 手动构建 TBN
                float cosA = _RotationSinCos.x;
                float sinA = _RotationSinCos.y;
                // [ cos  -sin ]
                // [ sin   cos ]
                half3 worldTangent = half3(cosA, sinA, 0);
                half3 worldBitangent = half3(-sinA, cosA, 0);
                half3 worldNormal = half3(0, 0, 1); 

                // 赋值
                OUT.tangentWS = worldTangent;
                OUT.bitangentWS = worldBitangent;
                OUT.normalWS = worldNormal;

                return OUT;
            }

            half4 Frag(Varyings IN) : SV_Target
            {
                half4 albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, IN.uv);
                
                half3 finalColor = albedo.rgb * IN.color;

                // 世界空间
                float2 posWS = posPixel2World(IN.positionCS.xy, _ScreenParams.xy);
                // 屏幕空间uv
                float2 screenUV = IN.positionCS.xy / _ScreenParams.xy;

                // 法线
                half4 packednorm = SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, IN.uv);
                half3 unpackednorm = UnpackNormal(packednorm);
                unpackednorm = normalize(unpackednorm);
                half3 normalWS = mul(half3x3(IN.tangentWS, IN.bitangentWS, IN.normalWS), unpackednorm);

                // RCWB GI
                bool isInsideSprite = false;
                RcwbLightData lightRCWBGI = GetRcwbLightData(screenUV,  _ScreenParams.xy, isInsideSprite);

                if (isInsideSprite && length(_Emission.rgb) > 0.0001f)
                {
                    lightRCWBGI.color = _Emission.rgb;
                }
                
                // 使用统一的兰伯特函数计算 RCWB GI 光照
                half3 realDirectionRCWBGI = normalize(half3(lightRCWBGI.direction.xy, 0.3));
                half lambertRCWBGI = CalculateLighting(normalWS, realDirectionRCWBGI);

                // SpotLight2D（带阴影和兰伯特）
                // fragmentZ = 0 表示片元在 Z=0 平面上
                float3 lightSpot = isInsideSprite ? CalculateAllSpotLights2D_Interior(posWS, normalWS) : CalculateAllSpotLights2D(posWS, normalWS, 0.0, true);

                half3 ansColor = albedo.xyz * lightRCWBGI.color * lambertRCWBGI + albedo.xyz * lightSpot;

                // debug:
                // ansColor = _Emission.rgb;
                // ansColor = isInsideSprite;
                
                return half4(ansColor, albedo.a * IN.color.a);
            }
            ENDHLSL
        }

    }
}