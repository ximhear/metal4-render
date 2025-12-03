# Metal4Render

Metal 4 API를 사용한 3D 삼각형 렌더링 데모 프로젝트입니다.

WWDC 2025에서 발표된 Metal 4의 새로운 기능들을 학습하고 테스트하기 위한 예제 코드입니다.

---

## 렌더링 결과

```
           빨강 (Red)
              ▲
             ╱ ╲
            ╱   ╲
           ╱     ╲
          ╱   ●   ╲  ← Y축을 중심으로 회전
         ╱         ╲
        ▼───────────▼
   파랑 (Blue)   초록 (Green)
```

- **왼손 좌표계** (Left-Hand Coordinate System)
- **원근 투영** (Perspective Projection)
- **RGB 그라디언트** 색상 보간
- **Y축 회전** 애니메이션 (60 FPS)

---

## Metal 4 소개 (WWDC 2025)

Metal 4는 Apple의 차세대 GPU 프로그래밍 API입니다.
기존 Metal API 대비 더 효율적인 GPU 리소스 관리와 향상된 성능을 제공합니다.

### Metal 4 vs 기존 Metal 비교

| 기능 | 기존 Metal | Metal 4 |
|------|------------|---------|
| **Command Queue** | `MTLCommandQueue` | `MTL4CommandQueue` |
| **Command Buffer** | 매 프레임 새로 생성 | 재사용 + `CommandAllocator` |
| **Resource Binding** | `setVertexBuffer()` | `ArgumentTable` + GPU Address |
| **Pipeline Creation** | `MTLDevice` | `MTL4Compiler` |
| **Memory Management** | 암시적 | `ResidencySet` (명시적) |
| **Synchronization** | `waitUntilCompleted` | `SharedEvent` + Signal |

### 주요 Metal 4 클래스

#### 1. MTL4CommandQueue
```swift
// Metal 4 커맨드 큐 생성
let commandQueue = device.makeMTL4CommandQueue()
```
더 효율적인 커맨드 제출과 동기화를 지원합니다.

#### 2. MTL4CommandBuffer + MTL4CommandAllocator
```swift
// 커맨드 버퍼 재사용
let commandBuffer = device.makeCommandBuffer()
let allocators = (0..<3).map { _ in device.makeCommandAllocator() }

// 매 프레임
allocator.reset()
commandBuffer.beginCommandBuffer(allocator: allocator)
// ... 렌더링 명령 ...
commandBuffer.endCommandBuffer()
```
커맨드 버퍼를 재사용하여 메모리 효율성을 향상시킵니다.

#### 3. MTL4ArgumentTable
```swift
// 기존 Metal
renderEncoder.setVertexBuffer(buffer, offset: 0, index: 0)

// Metal 4
let argumentTable = device.makeArgumentTable(descriptor: descriptor)
argumentTable.setAddress(buffer.gpuAddress, index: 0)
renderEncoder.setArgumentTable(argumentTable, stages: .vertex)
```
GPU 주소 기반의 유연한 리소스 바인딩을 제공합니다.

#### 4. MTLResidencySet
```swift
// GPU 메모리 관리
let residencySet = device.makeResidencySet(descriptor: descriptor)
residencySet.addAllocation(vertexBuffer)
residencySet.addAllocation(uniformBuffer)
residencySet.commit()
commandQueue.addResidencySet(residencySet)
```
명시적인 GPU 메모리 관리로 더 세밀한 리소스 수명 제어가 가능합니다.

#### 5. MTL4Compiler
```swift
// 파이프라인 상태 생성
let compiler = device.makeCompiler(descriptor: compilerDescriptor)
let pipelineState = try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
```
새로운 파이프라인 컴파일 시스템을 제공합니다.

---

## 프로젝트 구조

```
Metal4Render/
├── Metal4RenderApp.swift    # 앱 진입점 (@main)
├── ContentView.swift        # SwiftUI 메인 뷰
├── MetalView.swift          # SwiftUI-Metal 브릿지 뷰
├── Renderer.swift           # Metal 렌더링 엔진
│   ├── Metal4Renderer       # Metal 4 API 렌더러
│   └── LegacyMetalRenderer  # 기존 Metal API 렌더러 (폴백)
├── Shaders.metal            # GPU 셰이더 코드
└── Assets.xcassets/         # 앱 리소스
```

### 파일별 역할

| 파일 | 설명 |
|------|------|
| `Metal4RenderApp.swift` | SwiftUI 앱의 진입점. `@main` 속성으로 표시됨 |
| `ContentView.swift` | 메인 UI 뷰. `MetalView`를 전체 화면으로 표시 |
| `MetalView.swift` | `NSViewRepresentable`/`UIViewRepresentable`로 MTKView를 SwiftUI에 통합 |
| `Renderer.swift` | GPU 렌더링 로직. Metal 4 및 레거시 렌더러 모두 포함 |
| `Shaders.metal` | MSL(Metal Shading Language)로 작성된 GPU 셰이더 |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           앱 구조 개요                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────┐                                               │
│   │  Metal4RenderApp    │  ◄── @main 진입점                             │
│   │  (App Protocol)     │                                               │
│   └──────────┬──────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │    ContentView      │  ◄── SwiftUI 메인 뷰                          │
│   │   (SwiftUI View)    │                                               │
│   └──────────┬──────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │     MetalView       │  ◄── Metal 렌더링 뷰                          │
│   │ (NSView/UIView      │      (플랫폼별 Representable)                  │
│   │  Representable)     │                                               │
│   └──────────┬──────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │   Metal4Renderer    │  ◄── GPU 렌더링 로직                          │
│   │ or LegacyRenderer   │      (OS 버전에 따라 선택)                     │
│   └─────────────────────┘                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 렌더링 파이프라인

### GPU 렌더링 흐름

```
[Vertex Data]  ──▶  [Vertex Shader]  ──▶  [Rasterizer]  ──▶  [Fragment Shader]  ──▶  [Frame Buffer]
      │                    │                   │                     │
      │                    ▼                   ▼                     ▼
      │              클립 공간 좌표       프래그먼트 생성          최종 픽셀 색상
      │              (Clip Space)        (보간)
      │
      └─ 버텍스 위치 + 색상 데이터
```

### 좌표 변환 파이프라인

```
┌──────────┐     Model      ┌──────────┐     View       ┌──────────┐
│  Local   │ ─────────────▶ │  World   │ ─────────────▶ │   Eye    │
│  Space   │    Matrix      │  Space   │    Matrix      │  Space   │
└──────────┘                └──────────┘                └──────────┘
                                                              │
                                                              │ Projection
                                                              │ Matrix
                                                              ▼
                                                        ┌──────────┐
                                                        │   Clip   │
                                                        │  Space   │
                                                        └──────────┘
```

셰이더에서의 변환:
```metal
float4x4 mvp = projectionMatrix * viewMatrix * modelMatrix;
out.position = mvp * float4(vertex.position, 1.0);
```

---

## 좌표계 설명 (왼손 좌표계)

```
      +Y (위)
       │
       │
       │
       │
       └──────────── +X (오른쪽)
      ╱
     ╱
    ╱
   +Z (화면 안쪽, 멀어지는 방향)
```

**왼손 좌표계 특징:**
- X축 양의 방향: 오른쪽
- Y축 양의 방향: 위쪽
- Z축 양의 방향: 화면 안쪽 (멀어지는 방향)
- DirectX, Unity 등에서 사용하는 좌표계

---

## Triple Buffering

CPU와 GPU가 병렬로 작업할 수 있도록 3개의 프레임 버퍼를 사용합니다.

```
Frame 0: [Render] [Wait GPU] [        ]
Frame 1: [      ] [Render  ] [Wait GPU]
Frame 2: [      ] [        ] [Render  ]
Frame 3: [Render] [        ] [        ]  ← Frame 0의 GPU 완료 후 재사용
```

**동기화:**
```swift
// 프레임 N을 렌더링하려면 Frame N-3이 완료되어야 함
if frameIndex >= maxFramesInFlight {
    let valueToWait = frameIndex - maxFramesInFlight
    frameCompletionEvent.wait(untilSignaledValue: valueToWait, timeoutMS: 8)
}
```

---

## 지원 플랫폼

| 플랫폼 | Metal 4 지원 버전 | 폴백 렌더러 |
|--------|-------------------|-------------|
| macOS | 26.0+ | LegacyMetalRenderer |
| iOS | 26.0+ | LegacyMetalRenderer |
| iPadOS | 26.0+ | LegacyMetalRenderer |
| tvOS | 26.0+ | LegacyMetalRenderer |
| visionOS | 26.0+ | LegacyMetalRenderer |

---

## 빌드 요구사항

- **Xcode 26.0+**
- **Swift 6.0+**
- **macOS 26.0+ SDK** 또는 **iOS 26.0+ SDK**

### 빌드 및 실행

1. Xcode에서 `Metal4Render.xcodeproj` 열기
2. 타겟 디바이스 선택 (macOS, iOS Simulator, 또는 실제 디바이스)
3. `Cmd + R`로 빌드 및 실행

---

## 코드 구조 상세

### 버텍스 데이터 구조

```swift
struct Vertex {
    var position: SIMD3<Float>  // 3D 위치
    var color: SIMD4<Float>     // RGBA 색상
}
```

### 유니폼 데이터 구조

```swift
struct Uniforms {
    var modelMatrix: float4x4       // 모델 변환 (회전)
    var viewMatrix: float4x4        // 뷰 변환 (카메라)
    var projectionMatrix: float4x4  // 투영 변환 (원근)
}
```

### 삼각형 버텍스 정의

```swift
let vertices: [Vertex] = [
    Vertex(position: SIMD3<Float>(0.0, 0.5, 0.0),   // 상단 - 빨강
           color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0)),
    Vertex(position: SIMD3<Float>(0.5, -0.5, 0.0),  // 우하단 - 초록
           color: SIMD4<Float>(0.0, 1.0, 0.0, 1.0)),
    Vertex(position: SIMD3<Float>(-0.5, -0.5, 0.0), // 좌하단 - 파랑
           color: SIMD4<Float>(0.0, 0.0, 1.0, 1.0))
]
```

---

## 셰이더 코드

### 버텍스 셰이더

```metal
vertex VertexOut vertexShader(
    uint vertexID [[vertex_id]],
    const device Vertex* vertices [[buffer(0)]],
    const device Uniforms* uniforms [[buffer(1)]]
) {
    VertexOut out;
    float4x4 mvp = uniforms->projectionMatrix
                 * uniforms->viewMatrix
                 * uniforms->modelMatrix;
    out.position = mvp * float4(vertices[vertexID].position, 1.0);
    out.color = vertices[vertexID].color;
    return out;
}
```

### 프래그먼트 셰이더

```metal
fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    return in.color;  // 보간된 색상 출력
}
```

---

## 참고 자료

- [WWDC 2025 - What's New in Metal](https://developer.apple.com/wwdc25/)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

---

## 라이선스

이 프로젝트는 학습 목적으로 제작되었습니다.
자유롭게 사용, 수정, 배포할 수 있습니다.

---

## 변경 이력

### v1.0.0 (2025-12-04)
- Metal 4 기반 삼각형 렌더링 구현
- 왼손 좌표계 및 원근 투영 지원
- 레거시 Metal 렌더러 폴백 지원
- 상세한 한국어 주석 추가
