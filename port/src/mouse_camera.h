/* SPDX-License-Identifier: GPL-2.0-or-later */
/* inFAMOUS First Light (CUSA00897 1.00 eboot): the mouse turns the camera directly, as axylpup's
 * shadPS4 build does in Second Son 1.07 (src/core/mouse_aim.cpp there). The third-person camera's
 * update (the function at 0x171c00) stores the new pan and tilt, in radians, with
 *     vmovups %xmm0,0x330(%r15)      (r15 = the camera, xmm0 = pan, tilt, 0, 0)
 * at 0x17228a, the same store as Second Son's at 0x14ebce (r13 there). The jump there goes to a
 * cave that takes the mouse motion the pad thread left in a mailbox (xchg with 0), adds it, keeps
 * the tilt within the mailbox's limits (or where the game already had it) and does the store.
 * While `hold` is set (the mouse moved in the last seconds) the game's own turn of this update
 * is dropped: the angles start from the stored ones and the turn speeds stored next (0x170(%rsp),
 * loaded after the site) are zeroed. The game only turns the camera itself (it swings it behind
 * the player while gliding or running) when its right stick has not been used for a while; the
 * direct mouse never touches the stick. Every turn of the game's is dropped meanwhile, not only
 * that one: a scripted swing within those seconds is lost too (axylpup's build turns only the
 * follow off, with a byte not found in First Light yet). The gamepad's right stick ends the
 * hold (runtime_pad.c). Otherwise the store is the game's own, and the right stick keeps working
 * (it reads the angles back on the next update). rax and r11 are free at
 * the site (rax is loaded next, r11 is not used in the function); xmm2-xmm4 are kept on the
 * stack below the red zone.
 * Found by these bytes, so another build with the same code works too. */
#ifndef TLG_MOUSE_CAMERA_H
#define TLG_MOUSE_CAMERA_H
#include <stdint.h>
#include <string.h>

#define TLG_MOUSE_CAMERA_SITE_LENGTH 9u

/* The two stores before, the camera store and the speeds' load and store after it. */
static const unsigned char tlg_mouse_camera_code[36] = {
    0xc5, 0xf8, 0x11, 0x84, 0x24, 0x80, 0x01, 0x00, 0x00, /* vmovups %xmm0,0x180(%rsp) */
    0xc4, 0xc1, 0x78, 0x11, 0x87, 0x30, 0x03, 0x00, 0x00, /* vmovups %xmm0,0x330(%r15): the site */
    0xc5, 0xf8, 0x10, 0x8c, 0x24, 0x70, 0x01, 0x00, 0x00, /* vmovups 0x170(%rsp),%xmm1 */
    0xc4, 0xc1, 0x78, 0x11, 0x8f, 0x40, 0x03, 0x00, 0x00, /* vmovups %xmm1,0x340(%r15) */
};
#define TLG_MOUSE_CAMERA_SITE_IN_CODE 9u

/* Shared by the cave and the pad thread (runtime_pad.c). */
typedef struct {
    uint64_t delta;     /* +0: f32 pan, f32 tilt to add (radians); the cave swaps it with 0 */
    float pan, tilt;    /* +8: the angles the cave stored last */
    uint32_t updates;   /* +16: camera updates the cave saw */
    uint32_t hold;      /* +20: non-zero: the game's own turn is dropped (the pad thread sets it) */
    float tilt_low;     /* +24: the mouse keeps the tilt within these (radians) */
    float tilt_high;    /* +28 */
} TlgMouseCamera;

/* The site's address in the image (its offset from the image), or 0 when the code is not there
 * exactly once. */
static inline uint64_t tlg_mouse_camera_find(const unsigned char *image, uint64_t size) {
    uint64_t found = 0;
    for (uint64_t at = 0; at + sizeof(tlg_mouse_camera_code) <= size; ++at) {
        if (image[at] != tlg_mouse_camera_code[0] || memcmp(image + at, tlg_mouse_camera_code, sizeof(tlg_mouse_camera_code)))
            continue;
        if (found) return 0;
        found = at + TLG_MOUSE_CAMERA_SITE_IN_CODE;
    }
    return found;
}

/* The cave (llvm-mc, intel syntax): mailbox address at +2, return address at +176. */
static inline unsigned tlg_mouse_camera_build(unsigned char *out, uintptr_t mailbox, uintptr_t resume) {
    static const unsigned char code[184] = {
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, /* movabsq $mailbox, %rax */
        0x45, 0x31, 0xdb, /* xorl %r11d, %r11d */
        0x4c, 0x87, 0x18, /* xchgq %r11, (%rax) */
        0x83, 0x78, 0x14, 0x00, /* cmpl $0x0, 0x14(%rax) */
        0x74, 0x21, /* je 0x37 <motion> */
        0xc4, 0xc1, 0x78, 0x10, 0x87, 0x30, 0x03, 0x00, 0x00, /* vmovups 0x330(%r15), %xmm0 */
        0x48, 0xc7, 0x84, 0x24, 0x70, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* movq $0x0, 0x170(%rsp) */
        0x48, 0xc7, 0x84, 0x24, 0x78, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* movq $0x0, 0x178(%rsp) */
        0x4d, 0x85, 0xdb, /* testq %r11, %r11 */
        0x74, 0x5d, /* je 0x99 <store> */
        0x48, 0x8d, 0xa4, 0x24, 0x60, 0xff, 0xff, 0xff, /* leaq -0xa0(%rsp), %rsp */
        0xc5, 0xf8, 0x11, 0x14, 0x24, /* vmovups %xmm2, (%rsp) */
        0xc5, 0xf8, 0x11, 0x5c, 0x24, 0x10, /* vmovups %xmm3, 0x10(%rsp) */
        0xc5, 0xf8, 0x11, 0x64, 0x24, 0x20, /* vmovups %xmm4, 0x20(%rsp) */
        0xc4, 0xc1, 0xf9, 0x6e, 0xd3, /* vmovq %r11, %xmm2 */
        0xc5, 0xf8, 0x58, 0xda, /* vaddps %xmm2, %xmm0, %xmm3 */
        0xc4, 0xe2, 0x79, 0x18, 0x50, 0x18, /* vbroadcastss 0x18(%rax), %xmm2 */
        0xc5, 0xe8, 0x5d, 0xd0, /* vminps %xmm0, %xmm2, %xmm2 */
        0xc5, 0xe0, 0x5f, 0xd2, /* vmaxps %xmm2, %xmm3, %xmm2 */
        0xc4, 0xe2, 0x79, 0x18, 0x60, 0x1c, /* vbroadcastss 0x1c(%rax), %xmm4 */
        0xc5, 0xd8, 0x5f, 0xe0, /* vmaxps %xmm0, %xmm4, %xmm4 */
        0xc5, 0xe8, 0x5d, 0xd4, /* vminps %xmm4, %xmm2, %xmm2 */
        0xc4, 0xe3, 0x61, 0x0c, 0xc2, 0x02, /* vblendps $0x2, %xmm2, %xmm3, %xmm0 */
        0xc5, 0xf8, 0x10, 0x14, 0x24, /* vmovups (%rsp), %xmm2 */
        0xc5, 0xf8, 0x10, 0x5c, 0x24, 0x10, /* vmovups 0x10(%rsp), %xmm3 */
        0xc5, 0xf8, 0x10, 0x64, 0x24, 0x20, /* vmovups 0x20(%rsp), %xmm4 */
        0x48, 0x8d, 0xa4, 0x24, 0xa0, 0x00, 0x00, 0x00, /* leaq 0xa0(%rsp), %rsp */
        0xc5, 0xf9, 0xd6, 0x40, 0x08, /* vmovq %xmm0, 0x8(%rax) */
        0xff, 0x40, 0x10, /* incl 0x10(%rax) */
        0xc4, 0xc1, 0x78, 0x11, 0x87, 0x30, 0x03, 0x00, 0x00, /* vmovups %xmm0, 0x330(%r15) */
        0xff, 0x25, 0x00, 0x00, 0x00, 0x00, /* jmpq *(%rip) */
        0, 0, 0, 0, 0, 0, 0, 0, /* resume */
    };
    memcpy(out, code, sizeof(code));
    memcpy(out + 2, &mailbox, 8);
    memcpy(out + 176, &resume, 8);
    return sizeof(code);
}

/* The pad thread's side (runtime_pad.c); null until probe.c installed the cave. */
extern TlgMouseCamera *volatile tlg_mouse_camera;
#endif
