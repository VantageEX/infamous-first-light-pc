// Linux: the SHA-256 subset of BCrypt the renderer uses, over OpenSSL's libcrypto.
#pragma once
#include "iss_win_compat.h"
#include <openssl/evp.h>
using BCRYPT_ALG_HANDLE = void*;
using NTSTATUS = long;
#define BCRYPT_SHA256_ALGORITHM L"SHA256"
inline NTSTATUS BCryptOpenAlgorithmProvider(BCRYPT_ALG_HANDLE* alg, const wchar_t*, const wchar_t*, ULONG) {
    *alg = reinterpret_cast<void*>(1);
    return 0;
}
inline NTSTATUS BCryptHash(BCRYPT_ALG_HANDLE, unsigned char*, ULONG, unsigned char* data, ULONG size,
                           unsigned char* digest, ULONG digest_size) {
    unsigned int n = digest_size;
    return EVP_Digest(data, size, digest, &n, EVP_sha256(), nullptr) == 1 ? 0 : -1;
}
inline NTSTATUS BCryptCloseAlgorithmProvider(BCRYPT_ALG_HANDLE, ULONG) {
    return 0;
}
