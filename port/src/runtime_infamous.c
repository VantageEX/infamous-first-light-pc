/* System functions only inFAMOUS Second Son (CUSA00223 1.07) imports. All describe an offline
 * console with the whole game installed: PSN signed out (the "A" NpManager variants take a user
 * id instead of an online id, same answers as runtime_services.c), PlayGo with nothing left to
 * download, the player's own add-ons (ISS_ADDCONT_LABELS), networking calls that succeed without
 * a network. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"

#define NP_SIGNED_OUT ((int32_t)0x80550006)
#define APP_CONTENT_NOT_FOUND ((int32_t)0x80d90005)
#define SYSTEM_SERVICE_INVALID ((int32_t)0x80a10003)

static void once(const char *what) { printf("Runtime: %s (inFAMOUS)\n", what); }

/* sceSystemServiceParamGetString(id, buffer, size): only the user name parameter exists. */
static ABI int32_t param_string(int32_t id, char *buffer, uint64_t size) {
    if (!buffer || !size) return SYSTEM_SERVICE_INVALID;
    snprintf(buffer, size, "%s", "Player");
    printf("Runtime: sceSystemServiceParamGetString(%d) -> Player\n", id);
    return 0;
}
static ABI int32_t ok(void) { return 0; }
static ABI int32_t signed_out(void) { return NP_SIGNED_OUT; }
static ABI int32_t np_register_a(void *callback, void *arg) { (void)callback; (void)arg; static int32_t next = 0x100; return ++next; }
static ABI int32_t netctl_init(void) { once("sceNetCtlInit"); return 0; }
/* scePlayGoGetEta(handle, chunks, count, out_eta): everything is installed. */
static ABI int32_t playgo_eta(int32_t handle, const uint16_t *chunks, uint32_t count, uint64_t *eta) {
    (void)handle; (void)chunks; (void)count;
    if (eta) *eta = 0;
    return 0;
}
/* scePlayGoGetToDoList(handle, list, count, out_entries): no pending chunk. */
static ABI int32_t playgo_todo(int32_t handle, void *list, uint32_t count, uint32_t *entries) {
    (void)handle; (void)list; (void)count;
    if (entries) *entries = 0;
    return 0;
}
static ABI int32_t playgo_language_mask(int32_t handle, uint64_t mask) {
    printf("Runtime: scePlayGoSetLanguageMask(%#llx)\n", (unsigned long long)mask);
    (void)handle;
    return 0;
}
/* sceAppContentGetAddcontInfo(service, entitlement label, info). The game's add-ons (costumes,
 * Cole's Legacy) ship inside the base game and only need the entitlement: the labels listed in
 * ISS_ADDCONT_LABELS="LABEL,LABEL,..." (16 characters each, found by the launcher in the player's
 * own add-on packages or dumped add-on folders) are installed; any other is not. As in the
 * Castlesea inFAMOUS port (src/services.c content_addon, 8e167dc). */
static ABI int32_t addcont_info(uint32_t service, const char *label, uint8_t *info) {
    (void)service;
    if (!label || !info) return APP_CONTENT_NOT_FOUND;
    char want[17];
    memcpy(want, label, 16);
    want[16] = 0;
    const char *list = getenv("TLG_ADDCONT_LABELS");
    for (const char *p = list; p && *p;) {
        const char *end = strchr(p, ',');
        const size_t n = end ? (size_t)(end - p) : strlen(p);
        if (n == 16 && !memcmp(p, want, 16)) {
            /* SceAppContentAddcontInfo: the label (17 bytes, padded to 20), then the status. */
            const uint32_t installed = 4;
            memset(info, 0, 24);
            memcpy(info, want, 16);
            memcpy(info + 20, &installed, 4);
            printf("Runtime: add-on %s installed\n", want);
            return 0;
        }
        p = end ? end + 1 : NULL;
    }
    printf("Runtime: add-on %s not installed\n", want);
    return APP_CONTENT_NOT_FOUND;
}

static const RuntimeExport exports[] = {
    {"sceSystemServiceParamGetString", param_string},
    {"sceHttpSetChunkedTransferEnabled", ok}, {"sceHttpsLoadCert", ok},
    {"sceNetCtlInit", netctl_init}, {"sceNetCtlTerm", ok},
    {"scePlayGoGetEta", playgo_eta}, {"scePlayGoGetToDoList", playgo_todo},
    {"scePlayGoSetLanguageMask", playgo_language_mask},
    {"sceNpCheckNpAvailabilityA", signed_out}, {"sceNpGetAccountCountryA", signed_out},
    {"sceNpGetAccountIdA", signed_out}, {"sceNpGetGamePresenceStatusA", signed_out},
    {"sceNpRegisterGamePresenceCallbackA", np_register_a}, {"sceNpRegisterStateCallbackA", np_register_a},
    {"sceAppContentGetAddcontInfo", addcont_info},
};
uintptr_t runtime_infamous_resolve(const char *name) { return RUNTIME_LOOKUP(exports, name); }
