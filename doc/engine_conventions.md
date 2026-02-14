# i3 Engine - Conventions and Coordinate Systems

This document is the **single source of truth** for the engine's mathematical and coordinate system conventions.

---

## 1. Engine Standard (Logical Space)
The engine logic (Gameplay, Physics, Camera) operates in a unified space regardless of the backend.

- **Handedness:** **Right-Handed (RH)**.
- **Up Axis:** **+Y Up**.
- **Forward Axis:** **-Z Forward** (Standard GL/Mathematic convention).
- **Z Range:** **[0, 1]** (Zero to One).
    - 0 = Near Plane.
    - 1 = Far Plane.
    - *Note: This matches modern APIs (Vulkan, DX12) directly.*
- **Matrix Storage:** **Column-Major**.
    - $v' = P \cdot V \cdot M \cdot v$ (Pre-multiplication).
    - Memory: `[col0, col1, col2, col3]`.

---

## 2. Vulkan Target (Primary)
*Native Clip Space: Right-Handed, Y-Down, Z[0, 1].*

The engine (Y-Up) conflicts with Vulkan (Y-Down). We resolve this in the **Backend** (Rasterizer), keeping Matrices clean.

- **Adaptation Strategy:** **Negative Viewport**.
- **Viewport Config:**
    - `x` = 0, `width` = w
    - `y` = h, `height` = -h
    - `minDepth` = 0, `maxDepth` = 1
- **Winding Order:** **CCW (Counter-Clockwise)** is Front.
    - **Architectural Decision:** Do not expose `FrontFace` in the Pipeline API.
    - **Tradeoff:** We sacrifice "legacy asset compatibility" (rarely needed) for **Safety & Consistency**.
    - **Implementation:**
        - **Engine:** Always assumes CCW.
        - **Vulkan Backend:** Automatically sets `VK_FRONT_FACE_CLOCKWISE` to compensate for the Negative Viewport flip.
        - **DX12 Backend:** Automatically sets `D3D12_FRONT_FACE_COUNTER_CLOCKWISE`.
    - **User Control:** Only `CullMode` (`None`, `Back`, `Front`) is exposed.

---

## 3. DirectX 12 Target
*Native Clip Space: Right-Handed, Y-Up, Z[0, 1].*

DX12 matches our Logical Space almost perfectly.

- **Adaptation Strategy:** **Native / None**.
- **Viewport Config:**
    - `x` = 0, `width` = w
    - `y` = 0, `height` = h (Standard).
    - `minDepth` = 0, `maxDepth` = 1
- **Winding Order:** **Counter-Clockwise (CCW)** is Front.
    - No viewport flip means standard winding applies.

---

## 4. OpenGL Target (Low-End / Legacy)
*Native Clip Space: Right-Handed, Y-Up, Z[-1, 1].*

Legacy GL differs in Z-Range [-1, 1]. Modern GL (4.5+) resolves this.

- **Adaptation Strategy:** **glClipControl**.
    - We use `glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)` to force minimal Z-Range compliance.
- **Viewport Config:**
    - Standard GL Viewport (`y`=0 at bottom). 
    - No negative viewport support in older GL versions.
- **Winding Order:** **CCW** is Front.
- **Matrix Implication:**
    - If `glClipControl` is not available, the Projection Matrix **MUST** be patched to remap Z[0, 1] to Z[-1, 1] ($Z_{gl} = 2Z_{engine} - 1$). The engine prefers avoiding this path.

---

## 5. Transformation Cheat Sheet

| Feature | Engine Logic | Vulkan Backend | DX12 Backend | OpenGL (Modern) |
| :--- | :--- | :--- | :--- | :--- |
| **System** | Right-Handed | RH | RH | RH |
| **Up Axis** | +Y | -Y (Clip) | +Y (Clip) | +Y (Clip) |
| **Forward** | -Z | +Z (Clip) | +Z (Clip) | -Z (Clip) |
| **Z Range** | [0, 1] | [0, 1] | [0, 1] | [0, 1] (via ClipControl) |
| **Winding** | CCW (Fixed) | **CW** (Auto-Compensated) | **CCW** (Native) | **CCW** |
| **Viewport** | N/A | `height = -h` | `height = +h` | `height = +h` |
| **Matrices** | Column-Major | Column-Major | Column-Major | Column-Major |

### Shader Reference (Slang/HLSL)
```hlsl
// Conventions are handled by Backend State, not Shader Logic.
// Vertex Shader is identical for all backends.

struct VertexOutput {
    float4 pos : SV_Position;
};

VertexOutput main(float4 pos : POSITION, uniform float4x4 MVP) {
    VertexOutput output;
    // Standard Column-Major Multiply
    output.pos = mul(MVP, pos); 
    return output;
}

// Rasterizer State Configuration (Internal):
// VK:   FrontFace=CW (Hidden),   CullMode=User
// DX12: FrontFace=CCW (Hidden),  CullMode=User
```
```
