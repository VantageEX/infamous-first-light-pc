/* Which vibration calls reach the controller (src/rumble_gate.h): a game that sets the rumble
 * every frame sends at most 10 changes a second, a steady rumble is renewed, a stop always goes.
 * Time is passed in, so nothing sleeps. */
#include <assert.h>
#include <stdio.h>
#include "../src/rumble_gate.h"

#define MS(n) ((uint64_t)(n)*1000u)

int main(void) {
    RumbleGate g={0};
    uint64_t now=MS(1000);

    /* The first call goes; the same strength every frame does not, until the refresh is due. */
    assert(rumble_gate(&g,200,100,now));
    for (int i=0; i<29; ++i) assert(!rumble_gate(&g,200,100,now+=MS(16)));  /* 464 ms */
    assert(rumble_gate(&g,200,100,now+=MS(40)));                             /* 504 ms: renewed */

    /* A new strength every frame (draining with R2): one command per 100 ms, not 60 a second. */
    int sent=0;
    for (int i=0; i<60; ++i) sent+=rumble_gate(&g,(uint8_t)(100+i),50,now+=MS(16));
    assert(sent==8);                                                       /* every 7th frame */

    /* A stop goes at once, even right after a command; then nothing is resent while stopped. */
    assert(rumble_gate(&g,90,90,now+=MS(200)));
    assert(rumble_gate(&g,0,0,now+=MS(1)));
    assert(!rumble_gate(&g,0,0,now+=MS(600)));
    assert(!rumble_gate(&g,0,0,now+=MS(5000)));

    /* Starting again after a stop waits for the 100 ms like any change, then goes. */
    assert(rumble_gate(&g,255,255,now+=MS(100)));
    puts("rumble gate: ok");
    return 0;
}
