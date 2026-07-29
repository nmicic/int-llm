/* Host reference for the XIAO_RP2040_GPT harness.
 *
 * Loads model.mgw from disk into memory and drives the EXACT entry points
 * the firmware uses (mgpt_load_mem + mgpt_generate_sample), printing the
 * 20 samples and the FNV-1a 64 hash over their bytes. prepare.sh pins that
 * hash into include/gpt_pin.h; the on-target compare is then a genuine
 * cross-backend (host __int128 vs portable two-limb) and cross-ISA check
 * of the full inference path, PRNG stream included.
 *
 * Hash definition (must match src/main.cpp): FNV-1a 64 over each sample's
 * characters followed by one '\n', samples in order, standard offset/prime.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int mgpt_load_mem(const void *buf, size_t len);
int mgpt_generate_sample(char *out);

#define FNV64_OFFSET 1469598103934665603ULL
#define FNV64_PRIME  1099511628211ULL

static uint64_t fnv_sample(uint64_t h, const char *s) {
    for (; *s; s++) { h ^= (uint64_t)(unsigned char)*s; h *= FNV64_PRIME; }
    h ^= (uint64_t)'\n'; h *= FNV64_PRIME;
    return h;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "model.mgw";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 2; }
    long len = ftell(f);
    if (len <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 2; }
    void *buf = malloc((size_t)len);       /* malloc alignment >= 8 */
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "read error on %s\n", path);
        fclose(f);
        return 2;
    }
    fclose(f);

    if (mgpt_load_mem(buf, (size_t)len) != 0) {
        fprintf(stderr, "mgpt_load_mem rejected %s\n", path);
        return 2;
    }

    uint64_t h = FNV64_OFFSET;
    char sample[16];
    for (int i = 0; i < 20; i++) {
        mgpt_generate_sample(sample);
        printf("sample %2d: %s\n", i + 1, sample);
        h = fnv_sample(h, sample);
    }
    printf("GPT_SAMPLES_HASH=%016llx\n", (unsigned long long)h);
    free(buf);
    return 0;
}
