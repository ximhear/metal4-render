# Metal 3D 그래픽스 수학 가이드

Metal과 3D 그래픽스 프로그래밍에서 사용되는 수학적 개념들에 대한 종합 가이드입니다.

---

## 목차

1. [벡터 (Vectors)](#1-벡터-vectors)
2. [행렬 (Matrices)](#2-행렬-matrices)
3. [좌표계 (Coordinate Systems)](#3-좌표계-coordinate-systems)
4. [변환 행렬 (Transformation Matrices)](#4-변환-행렬-transformation-matrices)
5. [쿼터니언 (Quaternions)](#5-쿼터니언-quaternions)
6. [동차 좌표계 (Homogeneous Coordinates)](#6-동차-좌표계-homogeneous-coordinates)
7. [투영 (Projection)](#7-투영-projection)
8. [GPU에서의 수학 연산](#8-gpu에서의-수학-연산)

---

## 1. 벡터 (Vectors)

### 1.1 벡터란?

벡터는 **크기(magnitude)**와 **방향(direction)**을 가지는 수학적 객체입니다. 3D 그래픽스에서 벡터는 위치, 방향, 속도, 법선 등을 표현하는 데 사용됩니다.

```
        ↗ (2, 3)
       /
      /
     /
    ●───────→ x
    (원점)
```

### 1.2 SIMD 벡터 타입

Metal/Swift에서는 SIMD(Single Instruction, Multiple Data) 타입을 사용합니다:

| 타입 | 크기 | 용도 |
|------|------|------|
| `SIMD2<Float>` | 8 bytes | 텍스처 좌표 (u, v) |
| `SIMD3<Float>` | 16 bytes (정렬) | 3D 위치, 방향, 법선 |
| `SIMD4<Float>` | 16 bytes | 색상 (RGBA), 동차 좌표 |

```swift
// Swift에서의 벡터 선언
let position: SIMD3<Float> = SIMD3(1.0, 2.0, 3.0)
let color: SIMD4<Float> = SIMD4(1.0, 0.0, 0.0, 1.0)  // 빨간색
```

```metal
// MSL(Metal Shading Language)에서의 벡터
float3 position = float3(1.0, 2.0, 3.0);
float4 color = float4(1.0, 0.0, 0.0, 1.0);
```

### 1.3 벡터 연산

#### 덧셈과 뺄셈

```
벡터 덧셈: a + b = (a.x + b.x, a.y + b.y, a.z + b.z)
벡터 뺄셈: a - b = (a.x - b.x, a.y - b.y, a.z - b.z)
```

```swift
let a = SIMD3<Float>(1, 2, 3)
let b = SIMD3<Float>(4, 5, 6)
let sum = a + b        // (5, 7, 9)
let diff = b - a       // (3, 3, 3)
```

#### 스칼라 곱

```
스칼라 곱: k * v = (k * v.x, k * v.y, k * v.z)
```

```swift
let v = SIMD3<Float>(1, 2, 3)
let scaled = 2.0 * v   // (2, 4, 6)
```

#### 벡터 길이 (크기, Magnitude)

```
|v| = √(v.x² + v.y² + v.z²)
```

```swift
let v = SIMD3<Float>(3, 4, 0)
let len = length(v)    // 5.0 (피타고라스 정리: √(9+16) = 5)
```

#### 정규화 (Normalization)

단위 벡터(길이가 1인 벡터)를 만드는 과정:

```
normalize(v) = v / |v|
```

```swift
let v = SIMD3<Float>(3, 4, 0)
let n = normalize(v)   // (0.6, 0.8, 0) - 길이가 1
```

**왜 정규화가 필요한가?**
- 방향만 필요할 때 (조명 계산, 카메라 방향)
- 일관된 스케일의 연산을 위해
- 조명에서 법선 벡터는 항상 정규화되어야 함

### 1.4 내적 (Dot Product)

두 벡터의 내적은 **스칼라 값**을 반환합니다:

```
a · b = a.x*b.x + a.y*b.y + a.z*b.z
      = |a| * |b| * cos(θ)
```

여기서 θ는 두 벡터 사이의 각도입니다.

```
        b
        ↗
       /
      / θ
     /
    ●─────→ a
```

```swift
let a = SIMD3<Float>(1, 0, 0)
let b = SIMD3<Float>(0, 1, 0)
let d = dot(a, b)      // 0.0 (수직 = cos(90°) = 0)

let c = SIMD3<Float>(1, 0, 0)
let e = dot(a, c)      // 1.0 (평행 = cos(0°) = 1)
```

**내적의 활용:**

| 결과 | 의미 |
|------|------|
| `dot > 0` | 두 벡터가 같은 방향 (0° ~ 90°) |
| `dot = 0` | 두 벡터가 수직 (90°) |
| `dot < 0` | 두 벡터가 반대 방향 (90° ~ 180°) |

**그래픽스에서의 사용:**
- **조명 계산**: 표면 법선과 광원 방향의 내적 → 밝기
- **백페이스 컬링**: 시선 방향과 표면 법선의 내적 → 앞/뒤 판단
- **카메라 변환**: Look-at 행렬에서 이동량 계산

```swift
// Look-at 행렬에서의 내적 사용 예시
// 카메라 위치를 뷰 공간으로 변환
let translationX = -dot(xAxis, eye)
let translationY = -dot(yAxis, eye)
let translationZ = -dot(zAxis, eye)
```

### 1.5 외적 (Cross Product)

두 3D 벡터의 외적은 **두 벡터에 수직인 새로운 벡터**를 반환합니다:

```
a × b = (a.y*b.z - a.z*b.y,
         a.z*b.x - a.x*b.z,
         a.x*b.y - a.y*b.x)
```

```
        a × b (결과 벡터)
          ↑
          │
          │
          ●───→ a
         ╱
        ╱
       ↙ b
```

```swift
let a = SIMD3<Float>(1, 0, 0)  // X축
let b = SIMD3<Float>(0, 1, 0)  // Y축
let c = cross(a, b)            // (0, 0, 1) = Z축
```

**외적의 특성:**
- `|a × b| = |a| * |b| * sin(θ)` (평행사변형의 넓이)
- **비교환적**: `a × b = -(b × a)`
- 결과 벡터의 방향은 좌표계에 따라 다름 (왼손/오른손 법칙)

**그래픽스에서의 사용:**
- **법선 벡터 계산**: 삼각형의 두 변의 외적 → 표면 법선
- **좌표축 생성**: Look-at 행렬에서 카메라의 x, y, z 축 계산
- **회전축 찾기**: 두 방향 사이의 회전축

```swift
// Look-at 행렬에서 좌표축 계산
let zAxis = normalize(target - eye)     // 전방 벡터
let xAxis = normalize(cross(up, zAxis)) // 우측 벡터
let yAxis = cross(zAxis, xAxis)         // 상향 벡터
```

### 1.6 벡터 스위즐링 (Swizzling)

MSL에서는 벡터 컴포넌트에 유연하게 접근할 수 있습니다:

```metal
float4 color = float4(1.0, 0.5, 0.3, 1.0);

// 개별 컴포넌트 접근
float r = color.r;     // 1.0 (또는 color.x)
float g = color.g;     // 0.5 (또는 color.y)

// 스위즐링
float3 rgb = color.rgb;        // (1.0, 0.5, 0.3)
float3 bgr = color.bgr;        // (0.3, 0.5, 1.0) 순서 변경
float4 rrrr = color.rrrr;      // (1.0, 1.0, 1.0, 1.0) 복제
float2 xy = color.xy;          // (1.0, 0.5)
```

---

## 2. 행렬 (Matrices)

### 2.1 행렬이란?

행렬은 숫자들을 직사각형 형태로 배열한 것입니다. 3D 그래픽스에서는 주로 **4x4 행렬**을 사용하여 변환(이동, 회전, 크기조절)을 표현합니다.

```
4x4 행렬:
┌                     ┐
│  m00  m01  m02  m03 │
│  m10  m11  m12  m13 │
│  m20  m21  m22  m23 │
│  m30  m31  m32  m33 │
└                     ┘
```

### 2.2 행렬 저장 방식: Row-Major vs Column-Major

**중요**: Metal은 **Column-Major** 순서를 사용합니다!

```
행 기반 (Row-Major) - DirectX, 일부 수학 책:
메모리: [m00, m01, m02, m03, m10, m11, m12, m13, ...]

열 기반 (Column-Major) - Metal, OpenGL:
메모리: [m00, m10, m20, m30, m01, m11, m21, m31, ...]
```

Swift의 `float4x4`에서 열 접근:

```swift
// Column-Major: columns 파라미터로 생성
let matrix = float4x4(columns: (
    SIMD4<Float>(1, 0, 0, 0),  // 열 0
    SIMD4<Float>(0, 1, 0, 0),  // 열 1
    SIMD4<Float>(0, 0, 1, 0),  // 열 2
    SIMD4<Float>(0, 0, 0, 1)   // 열 3
))

// 열 접근
let column0 = matrix.columns.0  // 첫 번째 열
let column1 = matrix.columns.1  // 두 번째 열
```

### 2.3 단위 행렬 (Identity Matrix)

어떤 값에 곱해도 원래 값이 나오는 행렬:

```
I = ┌           ┐
    │ 1  0  0  0 │
    │ 0  1  0  0 │
    │ 0  0  1  0 │
    │ 0  0  0  1 │
    └           ┘

M × I = M
I × v = v
```

```swift
let identity = matrix_identity_float4x4
```

### 2.4 행렬 곱셈

두 행렬의 곱셈 결과의 각 요소는 행과 열의 내적입니다:

```
(A × B)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ
```

**행렬 곱셈은 비교환적입니다**: `A × B ≠ B × A`

```
변환 순서가 중요합니다!

먼저 회전(R), 그 다음 이동(T):
결과 = T × R × v

먼저 이동(T), 그 다음 회전(R):
결과 = R × T × v  ← 다른 결과!
```

```swift
let rotated_then_translated = translationMatrix * rotationMatrix * vertex
let translated_then_rotated = rotationMatrix * translationMatrix * vertex
// 두 결과는 다릅니다!
```

### 2.5 전치 행렬 (Transpose)

행과 열을 바꾼 행렬:

```
A = ┌       ┐       Aᵀ = ┌       ┐
    │ 1 2 3 │            │ 1 4 7 │
    │ 4 5 6 │            │ 2 5 8 │
    │ 7 8 9 │            │ 3 6 9 │
    └       ┘            └       ┘
```

```metal
// MSL에서
float4x4 transposed = transpose(matrix);
```

### 2.6 역행렬 (Inverse Matrix)

원래 행렬과 곱했을 때 단위 행렬이 되는 행렬:

```
M × M⁻¹ = I
M⁻¹ × M = I
```

**활용:**
- 뷰 행렬의 역 → 카메라의 월드 위치
- 모델 행렬의 역 → 월드 좌표를 로컬 좌표로

```swift
// Swift에서 역행렬 계산
let inverse = matrix.inverse
```

### 2.7 행렬식 (Determinant)

행렬의 스칼라 특성값:

```metal
// MSL에서
float det = determinant(matrix);
```

**활용:**
- `det = 0`: 역행렬이 존재하지 않음 (특이 행렬)
- `det < 0`: 행렬이 좌표계를 뒤집음 (반사 포함)
- `|det|`: 변환에 의한 부피 변화율

---

## 3. 좌표계 (Coordinate Systems)

### 3.1 왼손 좌표계 vs 오른손 좌표계

3D 공간에서 축의 방향을 정의하는 두 가지 규칙:

```
오른손 좌표계 (OpenGL 기본):        왼손 좌표계 (DirectX, Metal 기본):
        +Y                                  +Y
         │                                   │
         │                                   │
         │                                   │
         └────── +X                          └────── +X
        ╱                                     ╲
       ╱                                       ╲
      +Z (화면 밖으로)                          +Z (화면 안으로)

손가락으로 확인:
- 오른손: 엄지(+X), 검지(+Y), 중지(+Z)를 서로 수직으로
- 왼손: 엄지(+X), 검지(+Y), 중지(+Z)를 서로 수직으로
```

### 3.2 이 프로젝트의 좌표계

**Metal4Render는 왼손 좌표계를 사용합니다:**

```
        +Y (위)
         │
         │
         │
    ─────┼─────→ +X (오른쪽)
         │
         │
         ↓
        +Z (화면 안쪽, 멀어지는 방향)

카메라 위치: (0, 0, -3)  ← Z가 음수 = 화면 앞쪽
타겟 위치: (0, 0, 0)     ← 원점
카메라 시선: +Z 방향
```

### 3.3 회전 방향

**왼손 법칙에 따른 양의 회전 방향:**

```
Y축 회전 (위에서 내려다 볼 때):

        +Y
         │
         │    θ > 0
         │   ↻ (시계 방향)
         │
    ─────┼─────→ +X
         │
         ↓
        +Z

왼손으로 엄지를 Y축 양의 방향으로 향하면,
나머지 손가락이 감기는 방향이 양의 회전 방향
```

### 3.4 변환 공간들

3D 오브젝트가 화면에 그려지기까지 거치는 좌표 공간:

```
┌─────────────┐   Model    ┌─────────────┐   View     ┌─────────────┐
│ Local Space │ ─────────→ │ World Space │ ─────────→ │ View Space  │
│ (모델 좌표) │   Matrix   │ (월드 좌표) │   Matrix   │ (카메라 좌표)│
└─────────────┘            └─────────────┘            └─────────────┘
                                                            │
                                                            │ Projection
                                                            │ Matrix
                                                            ↓
┌─────────────┐  Viewport  ┌─────────────┐  ÷ w      ┌─────────────┐
│Screen Space │ ←───────── │    NDC      │ ←──────── │ Clip Space  │
│ (화면 좌표) │  Transform │ (-1 ~ +1)   │           │ (클립 좌표) │
└─────────────┘            └─────────────┘           └─────────────┘
```

| 공간 | 설명 | 좌표 범위 |
|------|------|-----------|
| **Local Space** | 모델 자체의 좌표계 (원점이 모델 중심) | 모델에 따라 다름 |
| **World Space** | 전체 씬의 좌표계 | 씬에 따라 다름 |
| **View Space** | 카메라 기준 좌표계 (카메라가 원점) | 씬에 따라 다름 |
| **Clip Space** | 투영 후 좌표 (w로 나누기 전) | [-w, w] |
| **NDC** | 정규화된 장치 좌표 | x,y: [-1, 1], z: [0, 1] |
| **Screen Space** | 실제 픽셀 좌표 | 화면 해상도 |

---

## 4. 변환 행렬 (Transformation Matrices)

### 4.1 이동 행렬 (Translation Matrix)

오브젝트를 (tx, ty, tz)만큼 이동:

```
T = ┌               ┐
    │ 1  0  0  tx   │
    │ 0  1  0  ty   │
    │ 0  0  1  tz   │
    │ 0  0  0  1    │
    └               ┘

T × [x, y, z, 1]ᵀ = [x+tx, y+ty, z+tz, 1]ᵀ
```

```swift
func translationMatrix(_ t: SIMD3<Float>) -> float4x4 {
    return float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(t.x, t.y, t.z, 1)
    ))
}
```

### 4.2 크기 행렬 (Scale Matrix)

오브젝트를 (sx, sy, sz)배로 확대/축소:

```
S = ┌               ┐
    │ sx  0   0   0 │
    │ 0   sy  0   0 │
    │ 0   0   sz  0 │
    │ 0   0   0   1 │
    └               ┘

S × [x, y, z, 1]ᵀ = [sx*x, sy*y, sz*z, 1]ᵀ
```

```swift
func scaleMatrix(_ s: SIMD3<Float>) -> float4x4 {
    return float4x4(columns: (
        SIMD4<Float>(s.x, 0, 0, 0),
        SIMD4<Float>(0, s.y, 0, 0),
        SIMD4<Float>(0, 0, s.z, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}
```

### 4.3 회전 행렬 (Rotation Matrices)

#### X축 회전

```
Rx = ┌                       ┐
     │ 1    0        0     0 │
     │ 0   cos(θ)  -sin(θ) 0 │
     │ 0   sin(θ)   cos(θ) 0 │
     │ 0    0        0     1 │
     └                       ┘
```

```swift
func rotationX(_ angle: Float) -> float4x4 {
    let c = cos(angle)
    let s = sin(angle)
    return float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, c, s, 0),
        SIMD4<Float>(0, -s, c, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}
```

#### Y축 회전 (이 프로젝트에서 사용)

```
Ry = ┌                       ┐
     │  cos(θ)  0   sin(θ)  0 │
     │    0     1     0     0 │
     │ -sin(θ)  0   cos(θ)  0 │
     │    0     0     0     1 │
     └                       ┘
```

```swift
// Renderer.swift에서 사용되는 Y축 회전 행렬
func rotationY(_ angle: Float) -> float4x4 {
    let c = cos(angle)
    let s = sin(angle)

    return float4x4(columns: (
        SIMD4<Float>(c, 0, -s, 0),    // 열 0
        SIMD4<Float>(0, 1, 0, 0),     // 열 1 (Y축은 변화 없음)
        SIMD4<Float>(s, 0, c, 0),     // 열 2
        SIMD4<Float>(0, 0, 0, 1)      // 열 3 (이동 없음)
    ))
}
```

#### Z축 회전

```
Rz = ┌                       ┐
     │ cos(θ)  -sin(θ)  0  0 │
     │ sin(θ)   cos(θ)  0  0 │
     │   0        0     1  0 │
     │   0        0     0  1 │
     └                       ┘
```

```swift
func rotationZ(_ angle: Float) -> float4x4 {
    let c = cos(angle)
    let s = sin(angle)
    return float4x4(columns: (
        SIMD4<Float>(c, s, 0, 0),
        SIMD4<Float>(-s, c, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}
```

### 4.4 복합 변환

변환들을 조합할 때는 **적용 순서의 역순**으로 행렬을 곱합니다:

```
원하는 순서: Scale → Rotate → Translate
행렬 곱셈: Transform = T × R × S

vertex' = T × R × S × vertex
        = T × (R × (S × vertex))
```

```swift
// 일반적인 모델 행렬 생성
let modelMatrix = translationMatrix(position) * rotationY(angle) * scaleMatrix(scale)
```

### 4.5 뷰 행렬 (View Matrix) - Look-At

카메라의 위치와 방향을 기반으로 월드 좌표를 뷰 좌표로 변환:

```
카메라 설정:
- eye: 카메라 위치
- target: 바라보는 지점
- up: 위쪽 방향

                      target (0, 0, 0)
                           ●
                          ╱│
                         ╱ │
                        ╱  │ up
                       ╱   │
                      ╱    │
                     ╱     │
          eye ●─────╱──────┘
       (0, 0, -3)
```

**좌표축 계산 (왼손 좌표계):**

```
1. zAxis (forward) = normalize(target - eye)
2. xAxis (right)   = normalize(cross(up, zAxis))
3. yAxis (up)      = cross(zAxis, xAxis)
```

**결과 행렬:**

```
View = ┌                                          ┐
       │ xAxis.x   yAxis.x   zAxis.x      0       │
       │ xAxis.y   yAxis.y   zAxis.y      0       │
       │ xAxis.z   yAxis.z   zAxis.z      0       │
       │ -dot(x,e) -dot(y,e) -dot(z,e)    1       │
       └                                          ┘

여기서 e = eye (카메라 위치)
```

```swift
// Renderer.swift의 lookAtLH 함수
func lookAtLH(eye: SIMD3<Float>,
              target: SIMD3<Float>,
              up: SIMD3<Float>) -> float4x4 {
    // 왼손 좌표계: 카메라는 +Z 방향을 바라봄
    let zAxis = normalize(target - eye)           // 전방 벡터
    let xAxis = normalize(cross(up, zAxis))       // 우측 벡터
    let yAxis = cross(zAxis, xAxis)               // 상향 벡터

    // 뷰 행렬 = 회전 × 이동 (카메라를 원점으로 이동)
    return float4x4(columns: (
        SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0),
        SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0),
        SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0),
        SIMD4<Float>(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1)
    ))
}
```

---

## 5. 쿼터니언 (Quaternions)

### 5.1 쿼터니언이란?

쿼터니언은 4개의 성분 `(x, y, z, w)` 또는 `(w, x, y, z)`로 구성된 수학적 객체로, 3D 회전을 표현하는 데 사용됩니다.

```
q = w + xi + yj + zk

여기서:
- w: 스칼라 부분 (실수)
- (x, y, z): 벡터 부분 (허수)
- i, j, k: 허수 단위 (i² = j² = k² = ijk = -1)
```

### 5.2 왜 쿼터니언을 사용하는가?

| 특성 | 회전 행렬 | 오일러 각 | 쿼터니언 |
|------|----------|----------|---------|
| 저장 공간 | 9~16개 float | 3개 float | 4개 float |
| 짐벌 락 | 없음 | **있음** | 없음 |
| 보간 | 어려움 | 어려움 | **쉬움 (Slerp)** |
| 연산 속도 | 느림 | 빠름 | 중간 |
| 누적 오차 | 있음 | 없음 | 적음 |

### 5.3 짐벌 락 (Gimbal Lock)

오일러 각의 치명적인 문제점:

```
짐벌 락 발생 상황 (Y축 90° 회전 시):

회전 전:                    Y축 90° 회전 후:
    Z                           Z
    │                           │
    │                           │
    └──X                   X────┘
   ╱                           (X축과 원래 Z축이 같은 평면!)
  Y

→ 자유도 상실: 3축 중 2축이 같은 평면에 놓여
  하나의 회전축을 잃어버림
```

**쿼터니언은 짐벌 락이 없습니다!**

### 5.4 단위 쿼터니언 (Unit Quaternion)

회전을 표현하려면 **단위 쿼터니언** (길이가 1)을 사용해야 합니다:

```
|q| = √(w² + x² + y² + z²) = 1
```

**축-각도(Axis-Angle)에서 쿼터니언 변환:**

```
회전축: (ax, ay, az) - 단위 벡터
회전각: θ

q = (cos(θ/2), ax*sin(θ/2), ay*sin(θ/2), az*sin(θ/2))
  = (w, x, y, z)
```

```swift
func quaternionFromAxisAngle(axis: SIMD3<Float>, angle: Float) -> simd_quatf {
    let halfAngle = angle * 0.5
    let s = sin(halfAngle)
    return simd_quatf(
        ix: axis.x * s,
        iy: axis.y * s,
        iz: axis.z * s,
        r: cos(halfAngle)
    )
}

// 또는 Swift SIMD 내장 함수 사용
let q = simd_quatf(angle: angle, axis: axis)
```

### 5.5 쿼터니언 연산

#### 곱셈 (회전 합성)

```swift
let q1 = simd_quatf(angle: .pi/4, axis: SIMD3(0, 1, 0))  // Y축 45°
let q2 = simd_quatf(angle: .pi/6, axis: SIMD3(1, 0, 0))  // X축 30°
let combined = q2 * q1  // q1 먼저, 그 다음 q2
```

#### 역쿼터니언 (역회전)

```swift
let inverse = q.inverse
// 또는 단위 쿼터니언의 경우: 켤레(conjugate)
let conjugate = q.conjugate  // (w, -x, -y, -z)
```

#### 벡터 회전

```swift
let v = SIMD3<Float>(1, 0, 0)
let rotatedV = q.act(v)  // q로 v를 회전
```

### 5.6 Slerp (Spherical Linear Interpolation)

두 회전 사이를 부드럽게 보간:

```
     q1 ●───────────────● q2
         \             /
          \    ●      /     ← Slerp(q1, q2, t)
           \  (t)    /
            \       /
             \     /
              \   /
               \ /
                ● 구의 중심
```

```swift
// t: 0.0 ~ 1.0 사이의 보간 계수
let interpolated = simd_slerp(q1, q2, t)

// 애니메이션 예시
for t in stride(from: 0.0, through: 1.0, by: 0.1) {
    let currentRotation = simd_slerp(startRotation, endRotation, Float(t))
    // 부드러운 회전 애니메이션
}
```

### 5.7 쿼터니언 ↔ 행렬 변환

```swift
// 쿼터니언 → 회전 행렬
let rotationMatrix = float4x4(q)

// 회전 행렬 → 쿼터니언
let q = simd_quatf(rotationMatrix)
```

**쿼터니언에서 4x4 회전 행렬 유도:**

```
q = (w, x, y, z)

R = ┌                                                ┐
    │ 1-2(y²+z²)    2(xy-wz)      2(xz+wy)      0   │
    │ 2(xy+wz)      1-2(x²+z²)    2(yz-wx)      0   │
    │ 2(xz-wy)      2(yz+wx)      1-2(x²+y²)    0   │
    │ 0             0             0             1   │
    └                                                ┘
```

---

## 6. 동차 좌표계 (Homogeneous Coordinates)

### 6.1 왜 동차 좌표를 사용하는가?

3D 좌표 `(x, y, z)`를 4D 좌표 `(x, y, z, w)`로 확장합니다.

**주요 이점:**
1. **이동(Translation)을 행렬 곱셈으로 표현** 가능
2. **모든 변환을 하나의 행렬로 통합** 가능
3. **투영 변환**을 자연스럽게 표현

```
3D 벡터로는 이동을 행렬 곱셈으로 표현할 수 없음:

┌       ┐   ┌   ┐     ┌       ┐
│ 1 0 0 │   │ x │     │   x   │
│ 0 1 0 │ × │ y │  =  │   y   │  ← 이동 없음!
│ 0 0 1 │   │ z │     │   z   │
└       ┘   └   ┘     └       ┘

4D 동차 좌표로는 가능:

┌           ┐   ┌   ┐     ┌       ┐
│ 1 0 0 tx  │   │ x │     │ x+tx  │
│ 0 1 0 ty  │ × │ y │  =  │ y+ty  │  ← 이동됨!
│ 0 0 1 tz  │   │ z │     │ z+tz  │
│ 0 0 0 1   │   │ 1 │     │   1   │
└           ┘   └   ┘     └       ┘
```

### 6.2 w 성분의 의미

| w 값 | 의미 | 용도 |
|------|------|------|
| `w = 1` | 점 (Position) | 3D 공간의 위치 |
| `w = 0` | 벡터 (Direction) | 방향 (이동의 영향을 받지 않음) |
| `w ≠ 1` | 투영 좌표 | 원근 분할 전 좌표 |

```swift
// 점 (위치) - 이동의 영향을 받음
let point = SIMD4<Float>(x, y, z, 1.0)

// 벡터 (방향) - 이동의 영향을 받지 않음
let direction = SIMD4<Float>(dx, dy, dz, 0.0)
```

```metal
// 셰이더에서 float3를 float4로 변환
float3 position = vertices[vertexID].position;
float4 homogeneous = float4(position, 1.0);  // w = 1.0 추가
```

### 6.3 원근 분할 (Perspective Division)

클립 좌표에서 NDC로 변환할 때 w로 나눕니다:

```
클립 좌표: (x, y, z, w)
    ↓ ÷ w
NDC: (x/w, y/w, z/w)
```

이것이 **원근감**을 만들어냅니다:
- 멀리 있는 물체는 w가 큼 → 나누면 작아짐 → 작게 보임
- 가까운 물체는 w가 작음 → 나누면 커짐 → 크게 보임

```
투영 전 (클립 공간):           투영 후 (NDC):

      ╱╲                           │  │
     ╱  ╲                          │  │
    ╱    ╲                         │  │
   ╱      ╲    w로 나눔            │  │
  ╱ 멀리   ╲  ───────────→        │작게│
 ╱          ╲                      │  │
╱   가까이   ╲                    │크게│
              (w 값 다름)          (균등한 공간)
```

---

## 7. 투영 (Projection)

### 7.1 투영이란?

3D 공간을 2D 화면에 표시하기 위한 변환입니다.

### 7.2 투영의 종류

```
원근 투영 (Perspective):         직교 투영 (Orthographic):

      ╱╲                              │    │
     ╱  ╲                             │    │
    ╱    ╲   시점                     │    │
   ╱      ╲ ●                         │    │
  ╱        ╲                          │    │
 ╱──────────╲                         │────│
   화면                                 화면

- 원근감 있음                      - 원근감 없음
- 게임, 시뮬레이션                 - CAD, 2D 게임, UI
```

### 7.3 원근 투영 행렬 (Perspective Projection)

**이 프로젝트에서 사용하는 왼손 좌표계 원근 투영:**

```swift
func perspectiveProjectionLH(fovYRadians fovy: Float,
                            aspect: Float,
                            nearZ: Float,
                            farZ: Float) -> float4x4 {
    // 수직 스케일: FOV가 클수록 더 많이 보임 (스케일 감소)
    let yScale = 1 / tan(fovy * 0.5)

    // 수평 스케일: 종횡비 보정
    let xScale = yScale / aspect

    // 깊이 범위
    let zRange = farZ - nearZ

    // Column-major 순서로 행렬 생성
    return float4x4(columns: (
        SIMD4<Float>(xScale, 0, 0, 0),                // 열 0: X 스케일
        SIMD4<Float>(0, yScale, 0, 0),                // 열 1: Y 스케일
        SIMD4<Float>(0, 0, farZ / zRange, 1),         // 열 2: Z 매핑 + W=Z
        SIMD4<Float>(0, 0, -nearZ * farZ / zRange, 0) // 열 3: Z 오프셋
    ))
}
```

**파라미터 설명:**

| 파라미터 | 설명 | 이 프로젝트의 값 |
|---------|------|-----------------|
| `fovy` | 수직 시야각 (라디안) | π/4 (45°) |
| `aspect` | 화면 가로/세로 비율 | 동적 계산 |
| `nearZ` | 근평면 거리 | 0.1 |
| `farZ` | 원평면 거리 | 100.0 |

### 7.4 시야각 (Field of View)

```
                    화면
                  ┌─────┐
                 /│     │\
                / │     │ \
               /  │     │  \
              /   │     │   \
             /    │     │    \
            / fov │     │     \
           /──────┼─────┼──────\
          ●       │     │
       카메라     │  d  │
                  └─────┘

tan(fov/2) = (화면 높이/2) / d

→ 작은 FOV: 좁은 시야, 망원 렌즈 효과
→ 큰 FOV: 넓은 시야, 광각 렌즈 효과, 왜곡 발생
```

### 7.5 클리핑 평면 (Clipping Planes)

```
        ┌──────────────────┐ Far Plane (z = farZ)
       ╱                  ╱
      ╱                  ╱
     ╱    View Frustum  ╱
    ╱    (절두체)       ╱
   ╱                  ╱
  ╱                  ╱
 └──────────────────┘ Near Plane (z = nearZ)
          ●
       카메라

- Near Plane 앞의 물체: 렌더링 안 됨
- Far Plane 뒤의 물체: 렌더링 안 됨
- 절두체 안의 물체만 렌더링
```

### 7.6 NDC (Normalized Device Coordinates)

**Metal의 NDC 범위:**

| 축 | 범위 | 설명 |
|----|------|------|
| X | [-1, +1] | 왼쪽 → 오른쪽 |
| Y | [-1, +1] | 아래 → 위 |
| Z | [0, 1] | Near → Far (Metal 특유!) |

**참고:** OpenGL은 Z가 [-1, +1] 범위입니다.

```
Metal NDC 공간:

        +Y (+1)
          │
          │
  (-1)────┼────(+1) +X
          │
          │
        -Y (-1)

Z: 0 (근평면) → 1 (원평면)
```

---

## 8. GPU에서의 수학 연산

### 8.1 정점 셰이더에서의 MVP 변환

```metal
vertex VertexOut vertexShader(
    uint vertexID [[vertex_id]],
    const device Vertex* vertices [[buffer(0)]],
    const device Uniforms* uniforms [[buffer(1)]]
) {
    VertexOut out;

    // MVP 행렬 계산: P × V × M
    float4x4 mvp = uniforms->projectionMatrix *
                   uniforms->viewMatrix *
                   uniforms->modelMatrix;

    // 정점 변환: 로컬 → 클립 좌표
    // float3를 float4로 변환 (w = 1.0)
    out.position = mvp * float4(vertices[vertexID].position, 1.0);

    out.color = vertices[vertexID].color;

    return out;
}
```

### 8.2 MSL 내장 수학 함수

#### 벡터 연산

| 함수 | 설명 | 예시 |
|------|------|------|
| `dot(a, b)` | 내적 | `float d = dot(normal, lightDir);` |
| `cross(a, b)` | 외적 (float3만) | `float3 n = cross(v1, v2);` |
| `length(v)` | 벡터 길이 | `float len = length(v);` |
| `distance(a, b)` | 두 점 사이 거리 | `float d = distance(p1, p2);` |
| `normalize(v)` | 정규화 | `float3 n = normalize(v);` |
| `reflect(I, N)` | 반사 벡터 | `float3 r = reflect(incident, normal);` |
| `refract(I, N, eta)` | 굴절 벡터 | `float3 r = refract(incident, normal, 1.5);` |

#### 행렬 연산

| 함수 | 설명 |
|------|------|
| `transpose(m)` | 전치 행렬 |
| `determinant(m)` | 행렬식 (2x2, 3x3, 4x4) |

**참고:** 역행렬은 MSL에 내장 함수가 없어 직접 구현해야 합니다.

#### 삼각 함수

| 함수 | 설명 |
|------|------|
| `sin(x)`, `cos(x)`, `tan(x)` | 기본 삼각 함수 |
| `asin(x)`, `acos(x)`, `atan(x)` | 역삼각 함수 |
| `atan2(y, x)` | 2인수 아크탄젠트 |
| `sincos(x, &s, &c)` | sin과 cos 동시 계산 (최적화) |

#### 유용한 수학 함수

| 함수 | 설명 |
|------|------|
| `abs(x)` | 절대값 |
| `sign(x)` | 부호 (-1, 0, 1) |
| `floor(x)`, `ceil(x)`, `round(x)` | 반올림 함수들 |
| `fract(x)` | 소수 부분: x - floor(x) |
| `min(a, b)`, `max(a, b)` | 최소/최대 |
| `clamp(x, min, max)` | 범위 제한 |
| `mix(a, b, t)` | 선형 보간: a*(1-t) + b*t |
| `smoothstep(e0, e1, x)` | 부드러운 보간 |
| `step(edge, x)` | x < edge ? 0 : 1 |
| `pow(x, y)` | 거듭제곱 |
| `sqrt(x)` | 제곱근 |
| `rsqrt(x)` | 역제곱근: 1/√x (빠름) |

### 8.3 정밀도 고려사항

```metal
// half (16비트) - 빠르지만 정밀도 낮음
half3 color = half3(1.0h, 0.5h, 0.0h);

// float (32비트) - 일반적으로 사용
float3 position = float3(1.0, 2.0, 3.0);

// 정밀도가 중요한 곳에서는 항상 float 사용
// - 위치 계산
// - 누적되는 값
// - 행렬 연산
```

### 8.4 성능 팁

```metal
// 나쁜 예: 불필요한 정규화
float3 v = normalize(someVector);
float3 v2 = normalize(v);  // 이미 정규화됨!

// 좋은 예
float3 v = normalize(someVector);
float3 v2 = v;  // 이미 단위 벡터

// 나쁜 예: 반복적인 length 계산
if (length(v) > 1.0) { ... }
if (length(v) < 0.5) { ... }

// 좋은 예: length_squared 사용 (sqrt 회피)
float lenSq = dot(v, v);  // length² = dot(v, v)
if (lenSq > 1.0) { ... }
if (lenSq < 0.25) { ... }  // 0.5² = 0.25

// 나쁜 예: 나눗셈
float result = a / b;

// 좋은 예: 곱셈으로 대체 (가능한 경우)
float invB = 1.0 / b;  // 한 번만 계산
float result = a * invB;
```

---

## 부록: 프로젝트 수학 코드 위치

이 프로젝트에서 수학 관련 코드가 있는 위치:

| 파일 | 함수/구조체 | 라인 | 설명 |
|------|------------|------|------|
| `Renderer.swift` | `perspectiveProjectionLH()` | 1548-1568 | 원근 투영 행렬 |
| `Renderer.swift` | `lookAtLH()` | 1616-1631 | 뷰 변환 행렬 |
| `Renderer.swift` | `rotationY()` | 1667-1677 | Y축 회전 행렬 |
| `Renderer.swift` | `Vertex` | 924-941 | 정점 구조체 |
| `Renderer.swift` | `Uniforms` | 972-990 | MVP 행렬 구조체 |
| `Shaders.metal` | `vertexShader()` | 822-875 | GPU 정점 변환 |
| `Shaders.metal` | 수학 함수 문서 | 484-516 | MSL 함수 레퍼런스 |

---

## 참고 자료

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [3D Math Primer for Graphics and Game Development](https://gamemath.com/)
- [Essential Mathematics for Games and Interactive Applications](https://www.essentialmath.com/)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
