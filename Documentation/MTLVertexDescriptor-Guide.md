# MTLVertexDescriptor 가이드

Metal에서 버텍스 데이터를 GPU에 전달하는 두 가지 방식과 MTLVertexDescriptor의 역할을 설명합니다.

---

## 목차

1. [버텍스 데이터 전달 방식](#1-버텍스-데이터-전달-방식)
2. [방식 1: 버퍼 직접 접근 (VertexDescriptor 없이)](#2-방식-1-버퍼-직접-접근-vertexdescriptor-없이)
3. [방식 2: MTLVertexDescriptor 사용](#3-방식-2-mtlvertexdescriptor-사용)
4. [어떤 방식을 선택해야 하는가?](#4-어떤-방식을-선택해야-하는가)
5. [MTLVertexDescriptor 상세 구조](#5-mtlvertexdescriptor-상세-구조)
6. [실전 예제](#6-실전-예제)

---

## 1. 버텍스 데이터 전달 방식

Metal에서 버텍스 셰이더에 데이터를 전달하는 방법은 크게 두 가지입니다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        버텍스 데이터 전달 방식                            │
├─────────────────────────────────┬───────────────────────────────────────┤
│     방식 1: 버퍼 직접 접근        │     방식 2: VertexDescriptor 사용      │
├─────────────────────────────────┼───────────────────────────────────────┤
│ • VertexDescriptor 불필요        │ • VertexDescriptor 필수               │
│ • 셰이더에서 포인터로 접근        │ • [[stage_in]] 어트리뷰트 사용         │
│ • 더 유연함                      │ • 더 선언적                           │
│ • 현재 프로젝트에서 사용          │ • 전통적인 방식                        │
└─────────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. 방식 1: 버퍼 직접 접근 (VertexDescriptor 없이)

### 이 프로젝트에서 사용하는 방식

MTLVertexDescriptor **없이** 버텍스 데이터를 전달합니다. 셰이더에서 직접 버퍼 포인터를 통해 데이터에 접근합니다.

### Swift 측 코드

```swift
// 버텍스 구조체 정의
struct Vertex {
    var position: SIMD3<Float>
    var color: SIMD4<Float>
}

// 버텍스 버퍼 생성
let vertices: [Vertex] = [
    Vertex(position: SIMD3(0, 0.5, 0), color: SIMD4(1, 0, 0, 1)),
    Vertex(position: SIMD3(0.5, -0.5, 0), color: SIMD4(0, 1, 0, 1)),
    Vertex(position: SIMD3(-0.5, -0.5, 0), color: SIMD4(0, 0, 1, 1))
]

vertexBuffer = device.makeBuffer(bytes: vertices,
                                  length: MemoryLayout<Vertex>.stride * vertices.count)

// 파이프라인 생성 - VertexDescriptor 설정 안 함!
let pipelineDescriptor = MTLRenderPipelineDescriptor()
pipelineDescriptor.vertexFunction = vertexFunction
pipelineDescriptor.fragmentFunction = fragmentFunction
// pipelineDescriptor.vertexDescriptor = nil  // 기본값, 설정 불필요

// 렌더링 시
renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
```

### Metal 셰이더 코드

```metal
// 셰이더에서 사용할 구조체 (Swift와 동일한 레이아웃)
struct Vertex {
    float3 position;
    float4 color;
};

// 버텍스 셰이더 - [[buffer(n)]]로 직접 접근
vertex VertexOut vertexShader(
    uint vertexID [[vertex_id]],                          // GPU가 제공하는 인덱스
    const device Vertex* vertices [[buffer(0)]],          // 버텍스 배열 포인터
    const device Uniforms* uniforms [[buffer(1)]]
) {
    VertexOut out;

    // vertexID를 사용해 직접 배열 인덱싱
    Vertex v = vertices[vertexID];

    out.position = uniforms->mvp * float4(v.position, 1.0);
    out.color = v.color;

    return out;
}
```

### 특징

| 장점 | 단점 |
|------|------|
| 간단하고 직관적 | 수동으로 인덱싱 관리 필요 |
| VertexDescriptor 설정 불필요 | 메모리 레이아웃을 직접 맞춰야 함 |
| 유연한 데이터 접근 가능 | 인터리브 외 레이아웃 시 복잡해질 수 있음 |
| 런타임에 동적 구조 가능 | |

---

## 3. 방식 2: MTLVertexDescriptor 사용

### [[stage_in]] 어트리뷰트와 함께 사용

MTLVertexDescriptor를 통해 버텍스 데이터의 레이아웃을 Metal에게 알려주고, 셰이더에서는 `[[stage_in]]`을 통해 자동으로 데이터를 받습니다.

### Swift 측 코드

```swift
// 버텍스 디스크립터 생성
let vertexDescriptor = MTLVertexDescriptor()

// Attribute 0: position (float3)
vertexDescriptor.attributes[0].format = .float3
vertexDescriptor.attributes[0].offset = 0
vertexDescriptor.attributes[0].bufferIndex = 0

// Attribute 1: color (float4)
vertexDescriptor.attributes[1].format = .float4
vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride  // 16 bytes (정렬됨)
vertexDescriptor.attributes[1].bufferIndex = 0

// Layout 0: 버퍼의 stride 설정
vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
vertexDescriptor.layouts[0].stepFunction = .perVertex
vertexDescriptor.layouts[0].stepRate = 1

// 파이프라인에 VertexDescriptor 설정 - 필수!
let pipelineDescriptor = MTLRenderPipelineDescriptor()
pipelineDescriptor.vertexFunction = vertexFunction
pipelineDescriptor.fragmentFunction = fragmentFunction
pipelineDescriptor.vertexDescriptor = vertexDescriptor  // ← 여기!
```

### Metal 셰이더 코드

```metal
// [[stage_in]]용 입력 구조체
// [[attribute(n)]]으로 VertexDescriptor의 attribute 인덱스와 매칭
struct VertexIn {
    float3 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

// 버텍스 셰이더 - [[stage_in]]으로 자동 데이터 수신
vertex VertexOut vertexShader(
    VertexIn in [[stage_in]],                             // Metal이 자동으로 채워줌
    constant Uniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;

    // vertexID나 배열 인덱싱 불필요!
    out.position = uniforms.mvp * float4(in.position, 1.0);
    out.color = in.color;

    return out;
}
```

### 특징

| 장점 | 단점 |
|------|------|
| Metal이 데이터 매핑을 처리 | 초기 설정이 복잡 |
| 명확한 데이터 레이아웃 정의 | 덜 유연함 |
| 여러 버퍼에서 어트리뷰트 조합 가능 | 구조체 레이아웃 변경 시 양쪽 수정 필요 |
| 인스턴스 데이터 쉽게 처리 | |

---

## 4. 어떤 방식을 선택해야 하는가?

### 결정 가이드

```
                    VertexDescriptor가 필요한가?
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
   [[stage_in]] 사용?                        [[buffer(n)]] 사용?
        │                                           │
        ▼                                           ▼
   ┌─────────┐                                ┌─────────┐
   │ 필요함   │                                │ 불필요  │
   └─────────┘                                └─────────┘
```

### VertexDescriptor가 **필요한** 경우

```swift
// 1. [[stage_in]]을 사용하는 셰이더
vertex VertexOut shader(VertexIn in [[stage_in]]) { ... }

// 2. 여러 버퍼에서 어트리뷰트를 조합할 때
vertexDescriptor.attributes[0].bufferIndex = 0  // 위치는 버퍼 0
vertexDescriptor.attributes[1].bufferIndex = 1  // 색상은 버퍼 1
vertexDescriptor.attributes[2].bufferIndex = 2  // 노말은 버퍼 2

// 3. 인스턴스 데이터를 사용할 때
vertexDescriptor.layouts[1].stepFunction = .perInstance
vertexDescriptor.layouts[1].stepRate = 1
```

### VertexDescriptor가 **불필요한** 경우

```swift
// 1. [[buffer(n)]]로 직접 접근하는 셰이더 (이 프로젝트)
vertex VertexOut shader(
    uint vid [[vertex_id]],
    const device Vertex* vertices [[buffer(0)]]
) { ... }

// 2. Compute 셰이더 (버텍스 셰이더가 아님)
kernel void compute(device float* data [[buffer(0)]]) { ... }

// 3. 구조화되지 않은 데이터 접근
// 예: 파티클 시스템에서 동적 버텍스 수
```

### 비교표

| 상황 | 권장 방식 |
|------|----------|
| 간단한 메시 렌더링 | 버퍼 직접 접근 (이 프로젝트) |
| 복잡한 버텍스 포맷 | VertexDescriptor |
| 여러 버퍼 조합 | VertexDescriptor |
| 인스턴싱 | VertexDescriptor |
| 동적 버텍스 구조 | 버퍼 직접 접근 |
| Compute 셰이더 | 버퍼 직접 접근 (해당 없음) |

---

## 5. MTLVertexDescriptor 상세 구조

### 전체 구조

```
MTLVertexDescriptor
├── attributes[0...30]     ← 최대 31개의 버텍스 어트리뷰트
│   ├── format            ← 데이터 타입 (float3, half4, etc.)
│   ├── offset            ← 버텍스 내 바이트 오프셋
│   └── bufferIndex       ← 어떤 버퍼에서 읽을지 (0-30)
│
└── layouts[0...30]        ← 최대 31개의 버퍼 레이아웃
    ├── stride            ← 버텍스 간 바이트 간격
    ├── stepFunction      ← perVertex / perInstance / constant
    └── stepRate          ← 몇 버텍스/인스턴스마다 진행할지
```

### Attribute Formats (주요 포맷)

```swift
// 정수 타입
.uchar, .uchar2, .uchar3, .uchar4           // 8-bit unsigned
.char, .char2, .char3, .char4               // 8-bit signed
.ushort, .ushort2, .ushort3, .ushort4       // 16-bit unsigned
.short, .short2, .short3, .short4           // 16-bit signed
.uint, .uint2, .uint3, .uint4               // 32-bit unsigned
.int, .int2, .int3, .int4                   // 32-bit signed

// 부동소수점 타입
.half, .half2, .half3, .half4               // 16-bit float
.float, .float2, .float3, .float4           // 32-bit float

// 정규화 타입 (0.0-1.0 또는 -1.0-1.0으로 변환)
.ucharNormalized, .uchar4Normalized         // 0-255 → 0.0-1.0
.shortNormalized, .short4Normalized         // -32768~32767 → -1.0~1.0

// 패킹 타입
.uint1010102Normalized    // 10-10-10-2 비트 패킹 (노말 압축에 유용)
```

### Step Functions

```swift
// 버텍스마다 데이터 진행 (기본값)
layouts[0].stepFunction = .perVertex
layouts[0].stepRate = 1

// 인스턴스마다 데이터 진행
layouts[1].stepFunction = .perInstance
layouts[1].stepRate = 1

// 상수 (모든 버텍스/인스턴스에 동일)
layouts[2].stepFunction = .constant
layouts[2].stepRate = 0
```

---

## 6. 실전 예제

### 예제 1: 기본 메시 (인터리브 레이아웃)

가장 일반적인 패턴 - 모든 어트리뷰트가 하나의 버퍼에 인터리브됨:

```swift
// 메모리 레이아웃:
// [pos0|normal0|uv0][pos1|normal1|uv1][pos2|normal2|uv2]...

struct Vertex {
    var position: SIMD3<Float>   // offset: 0
    var normal: SIMD3<Float>     // offset: 16 (SIMD3는 16바이트 정렬)
    var texCoord: SIMD2<Float>   // offset: 32
}
// stride = 48 bytes (16 + 16 + 8 + 8 padding)

let descriptor = MTLVertexDescriptor()

// Position
descriptor.attributes[0].format = .float3
descriptor.attributes[0].offset = 0
descriptor.attributes[0].bufferIndex = 0

// Normal
descriptor.attributes[1].format = .float3
descriptor.attributes[1].offset = 16
descriptor.attributes[1].bufferIndex = 0

// TexCoord
descriptor.attributes[2].format = .float2
descriptor.attributes[2].offset = 32
descriptor.attributes[2].bufferIndex = 0

// Buffer layout
descriptor.layouts[0].stride = 48
descriptor.layouts[0].stepFunction = .perVertex
```

```metal
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 texCoord [[attribute(2)]];
};

vertex VertexOut vertexShader(VertexIn in [[stage_in]]) {
    // ...
}
```

### 예제 2: 분리 버퍼 (Separate Buffers)

어트리뷰트별로 다른 버퍼 사용:

```swift
// 버퍼 0: 위치만
// 버퍼 1: 노말만
// 버퍼 2: UV만

let descriptor = MTLVertexDescriptor()

// Position from buffer 0
descriptor.attributes[0].format = .float3
descriptor.attributes[0].offset = 0
descriptor.attributes[0].bufferIndex = 0
descriptor.layouts[0].stride = MemoryLayout<SIMD3<Float>>.stride

// Normal from buffer 1
descriptor.attributes[1].format = .float3
descriptor.attributes[1].offset = 0
descriptor.attributes[1].bufferIndex = 1
descriptor.layouts[1].stride = MemoryLayout<SIMD3<Float>>.stride

// TexCoord from buffer 2
descriptor.attributes[2].format = .float2
descriptor.attributes[2].offset = 0
descriptor.attributes[2].bufferIndex = 2
descriptor.layouts[2].stride = MemoryLayout<SIMD2<Float>>.stride

// 렌더링 시 각각 바인딩
renderEncoder.setVertexBuffer(positionBuffer, offset: 0, index: 0)
renderEncoder.setVertexBuffer(normalBuffer, offset: 0, index: 1)
renderEncoder.setVertexBuffer(uvBuffer, offset: 0, index: 2)
```

### 예제 3: 인스턴싱 (Instancing)

같은 메시를 여러 번 그릴 때:

```swift
struct InstanceData {
    var modelMatrix: float4x4
    var color: SIMD4<Float>
}

let descriptor = MTLVertexDescriptor()

// Per-vertex: position (buffer 0)
descriptor.attributes[0].format = .float3
descriptor.attributes[0].offset = 0
descriptor.attributes[0].bufferIndex = 0
descriptor.layouts[0].stride = MemoryLayout<SIMD3<Float>>.stride
descriptor.layouts[0].stepFunction = .perVertex

// Per-instance: model matrix (buffer 1, 4개의 float4로 분할)
for i in 0..<4 {
    descriptor.attributes[1 + i].format = .float4
    descriptor.attributes[1 + i].offset = i * 16
    descriptor.attributes[1 + i].bufferIndex = 1
}

// Per-instance: color (buffer 1)
descriptor.attributes[5].format = .float4
descriptor.attributes[5].offset = 64
descriptor.attributes[5].bufferIndex = 1

descriptor.layouts[1].stride = MemoryLayout<InstanceData>.stride
descriptor.layouts[1].stepFunction = .perInstance  // ← 핵심!
descriptor.layouts[1].stepRate = 1

// 인스턴스 렌더링
renderEncoder.drawIndexedPrimitives(
    type: .triangle,
    indexCount: indexCount,
    indexType: .uint16,
    indexBuffer: indexBuffer,
    indexBufferOffset: 0,
    instanceCount: 100  // 100개 인스턴스 그리기
)
```

```metal
struct VertexIn {
    float3 position [[attribute(0)]];
    float4 modelMatrix0 [[attribute(1)]];
    float4 modelMatrix1 [[attribute(2)]];
    float4 modelMatrix2 [[attribute(3)]];
    float4 modelMatrix3 [[attribute(4)]];
    float4 instanceColor [[attribute(5)]];
};

vertex VertexOut vertexShader(VertexIn in [[stage_in]]) {
    float4x4 modelMatrix = float4x4(
        in.modelMatrix0,
        in.modelMatrix1,
        in.modelMatrix2,
        in.modelMatrix3
    );
    // ...
}
```

---

## 7. 이 프로젝트의 선택 이유

### 현재 프로젝트: 버퍼 직접 접근 방식

```swift
// Renderer.swift - VertexDescriptor 없이 파이프라인 생성
let pipelineDescriptor = MTL4RenderPipelineDescriptor()
pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDesc
pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDesc
// vertexDescriptor 설정 없음!
```

```metal
// Shaders.metal - [[buffer(n)]]으로 직접 접근
vertex VertexOut vertexShader(
    uint vertexID [[vertex_id]],
    const device Vertex* vertices [[buffer(0)]],
    const device Uniforms* uniforms [[buffer(1)]]
) {
    Vertex v = vertices[vertexID];
    // ...
}
```

### 이 방식을 선택한 이유

1. **단순함**: 삼각형 하나만 그리는 간단한 예제
2. **Metal 4 스타일**: GPU 주소 직접 사용과 잘 어울림
3. **유연성**: 동적 버텍스 구조에 대응 가능
4. **학습 목적**: 기본 개념 이해에 집중

### VertexDescriptor로 전환하려면

```swift
// 1. VertexDescriptor 생성
let vertexDescriptor = MTLVertexDescriptor()
vertexDescriptor.attributes[0].format = .float3
vertexDescriptor.attributes[0].offset = 0
vertexDescriptor.attributes[0].bufferIndex = 0
vertexDescriptor.attributes[1].format = .float4
vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
vertexDescriptor.attributes[1].bufferIndex = 0
vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride

// 2. 파이프라인에 설정
pipelineDescriptor.vertexDescriptor = vertexDescriptor

// 3. 셰이더 수정
// [[buffer(0)]] → [[stage_in]]
```

---

## 요약

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              요약                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VertexDescriptor 필요:                                                 │
│  • [[stage_in]] 사용 시                                                 │
│  • 여러 버퍼에서 어트리뷰트 조합 시                                       │
│  • 인스턴싱 사용 시                                                      │
│                                                                         │
│  VertexDescriptor 불필요:                                               │
│  • [[buffer(n)]] + [[vertex_id]]로 직접 접근 시 (이 프로젝트)            │
│  • Compute 셰이더                                                       │
│  • 동적/유연한 버텍스 구조 필요 시                                        │
│                                                                         │
│  핵심 차이:                                                              │
│  • [[stage_in]]: Metal이 데이터를 자동으로 구조체에 채워줌                │
│  • [[buffer(n)]]: 개발자가 직접 포인터로 접근                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
