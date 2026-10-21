/* The mouse through the pad (src/runtime_pad.c): camera on the right stick, buttons as PS4
 * buttons, and touchpad mode (hold the mode key, here Mouse X1): the mouse becomes a finger. The
 * window's mouse is a stub the test drives; scePadReadState runs for real. */
#define _GNU_SOURCE
#include <assert.h>
#include <unistd.h>
#include "../src/runtime_pad.c"

static TlgGpuMouse fake;
static TlgGpuInputSettings settings;
static int hud_active, hud_touch, hud_click;
static float hud_x, hud_y;
void tlggpu_mouse_take(TlgGpuMouse *mouse) {
    *mouse=fake;
    fake.dx=fake.dy=0; fake.pressed=0;
    fake.wheel_up=fake.wheel_down=fake.wheel_left=fake.wheel_right=0;
}
void tlggpu_input_settings(TlgGpuInputSettings *out) { *out=settings; }
void tlggpu_mouse_pad_hud(int active, float x, float y, int touching, int clicking) {
    hud_active=active; hud_x=x; hud_y=y; hud_touch=touching; hud_click=clicking;
}
uint64_t tlggpu_total_draws(void) { return 0; }
void tlggpu_frame_counters(TlgGpuFrameCounters *out) { memset(out,0,sizeof(*out)); }
int tlggpu_overlay_captures_input(void) { return 0; }
void tlggpu_overlay_notice(const char *text, float seconds) { (void)text; (void)seconds; }
void tlggpu_graffiti_guide(int force) { (void)force; }
float tlggpu_sensor_aim_scale(void) { return 1.0f; }
uintptr_t runtime_lookup(const RuntimeExport *table, size_t count, const char *name) {
    (void)table; (void)count; (void)name;
    return 0;
}

enum { LEFT=1, MIDDLE=2, RIGHT=4, X1=8, X2=16 }; /* 1 << (SDL button - 1) */
static PadData read_pad(void) {
    PadData d;
    assert(pad_read_state(1,&d)==0);
    return d;
}
static void hold(uint32_t buttons) { fake.held=buttons; fake.pressed=buttons; }
/* A few samples of steady motion: counts per sample at ~50 Hz. */
static PadData move(float dx, float dy, int samples) {
    PadData d={0};
    for (int i=0;i<samples;++i) { fake.dx=dx; fake.dy=dy; usleep(20000); d=read_pad(); }
    return d;
}

int main(void) {
    setenv("SDL_VIDEODRIVER","dummy",1);
    SDL_SetHint(SDL_HINT_GAMECONTROLLER_IGNORE_DEVICES_EXCEPT,"0x1d50/0x6189");
    assert(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_GAMEPAD));
    assert(pad_init()==0 && pad_open(1,0,0,NULL)==1);
    settings=(TlgGpuInputSettings){.mouse_enabled=1,.sensitivity=1.0f,.smoothing=0.0f};
    fake.captured=1;

    /* Camera: motion is the right stick, up/down and left/right with the right signs. */
    PadData d=move(0,0,2);
    assert(d.right_x==128 && d.right_y==128);
    d=move(60,0,3);
    assert(d.right_x>200 && d.right_y==128);
    d=move(-60,0,3);
    assert(d.right_x<56);
    d=move(0,60,3);
    assert(d.right_y>200);
    settings.invert_y=1;
    d=move(0,60,3);
    assert(d.right_y<56);
    settings.invert_y=0;
    d=move(0,0,12);
    assert(d.right_x==128 && d.right_y==128); /* at rest the stick is exactly centred */
    /* Not captured (Esc, menu, focus lost): nothing. */
    fake.captured=0;
    d=move(60,0,3);
    assert(d.right_x==128);
    fake.captured=1;
    settings.mouse_enabled=0;
    d=move(60,0,3);
    assert(d.right_x==128);
    settings.mouse_enabled=1;
    move(0,0,3);

    /* Buttons: left R2, right L2, middle R3; the analog triggers read fully pressed. */
    hold(LEFT); d=move(0,0,1);
    assert((d.buttons & BTN_R2) && d.r2==255 && !(d.buttons & (BTN_L2|BTN_R3)));
    hold(RIGHT); d=move(0,0,1);
    assert((d.buttons & BTN_L2) && d.l2==255 && !(d.buttons & BTN_R2));
    hold(MIDDLE); d=move(0,0,1);
    assert((d.buttons & BTN_R3) && !(d.buttons & (BTN_L2|BTN_R2)));
    hold(0); d=move(0,0,1);
    assert(!(d.buttons & (BTN_L2|BTN_R2|BTN_R3)) && d.touch_count==0);
    /* A click shorter than a sample (pressed, already up) is not lost. */
    fake.pressed=LEFT; d=read_pad();
    assert(d.buttons & BTN_R2);

    /* Touchpad mode: X1 held. The camera stops, the buttons are fingers. */
    hold(X1); d=move(60,0,3);
    assert(d.right_x==128 && d.touch_count==0 && !(d.buttons & (BTN_R2|BTN_L2|BTN_R3)));
    assert(hud_active && !hud_touch);
    assert(hud_x>0.5f); /* the hovering finger moved right of the middle */
    /* Left down: the finger lands where it is; the click comes after the lead. */
    hold(X1|LEFT); d=move(0,0,1);
    assert(d.touch_count==1 && !(d.buttons & BTN_TOUCHPAD) && !(d.buttons & BTN_R2));
    const uint16_t x0=d.touches[0].x, y0=d.touches[0].y;
    const uint8_t id0=d.touches[0].id;
    assert(x0>959 && x0<=1919 && y0==471);
    assert(hud_touch);
    usleep(60000);
    d=move(0,0,1);
    assert(d.touch_count==1 && (d.buttons & BTN_TOUCHPAD) && d.touches[0].id==id0 && hud_click);
    /* Drag right: the click is released and the finger slides on (Open Door). */
    d=move(20,0,1); /* 32 units: under the 80 of TLG_MOUSE_PAD_DRAG, still a click */
    assert(d.touch_count==1 && (d.buttons & BTN_TOUCHPAD) && d.touches[0].x>x0+20);
    d=move(60,0,1); /* 96 more units: a swipe */
    assert(d.touch_count==1 && !(d.buttons & BTN_TOUCHPAD) && d.touches[0].x>x0+100 && d.touches[0].id==id0);
    d=move(30,0,1);
    assert(d.touches[0].x>x0+150 && d.touches[0].y==y0 && !(d.buttons & BTN_TOUCHPAD));
    /* Release: the finger lingers, then lifts. */
    hold(X1); d=move(0,0,1);
    assert(d.touch_count==1 && !(d.buttons & BTN_TOUCHPAD));
    usleep(150000);
    d=move(0,0,1);
    assert(d.touch_count==0 && d.buttons==0);

    /* Middle: a finger without a click, however long. */
    hold(X1|MIDDLE); d=move(0,0,1);
    assert(d.touch_count==1);
    usleep(100000);
    d=move(0,0,1);
    assert(d.touch_count==1 && !(d.buttons & BTN_TOUCHPAD) && !(d.buttons & BTN_R3));
    hold(X1); move(0,0,1);
    usleep(150000);
    d=move(0,0,1);
    assert(d.touch_count==0);

    /* Right: two fingers at both ends of the pad and the pad pressed (the car off Reggie). */
    hold(X1|RIGHT); d=move(0,0,1);
    assert(d.touch_count==2 && d.touches[0].x==120 && d.touches[1].x==1800 && !(d.buttons & BTN_L2));
    usleep(60000);
    d=move(0,0,1);
    assert(d.touch_count==2 && (d.buttons & BTN_TOUCHPAD) && !(d.buttons & BTN_L2));
    hold(X1); move(0,0,1);
    usleep(150000);
    d=move(0,0,1);
    assert(d.touch_count==0 && d.buttons==0);

    /* Leaving the mode: back to the camera, left is R2 again. */
    hold(0); d=move(0,0,2);
    assert(!hud_active);
    d=move(60,0,3);
    assert(d.right_x>200);
    hold(LEFT); d=move(0,0,1);
    assert(d.buttons & BTN_R2);
    puts("PASS: mouse through the pad");
    return 0;
}
