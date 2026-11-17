package vp8enc

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	"image/draw"
	"os"
	"sort"
	"strings"
	"testing"
)

// The encoder output must not depend on the build: the assembly kernels
// are bit-exact with the Go code, so every configuration below must encode
// to the bytes the pure-Go encoder produced before the kernels existed.
// identityHashes holds the SHA-256 (first 16 hex digits) of each case's
// output as produced at b6b0954 (pure Go, before the NEON kernels);
// TestEncodeIdentity checks them in every build (go test, go test -tags
// purego, other architectures). Set VP8ENC_PRINT_HASHES=1 to print the
// current hashes in the format of the table.

type identityCase struct {
	name  string
	img   string // test image name, or "synth<kind>"
	w, h  int    // crop / synthetic size
	sharp bool
	exact bool
	cfg   Config
}

func identityCases() []identityCase {
	var cases []identityCase
	add := func(img string, w, h int, sharp, exact bool, name string, mod func(*Config)) {
		cfg := DefaultConfig()
		mod(&cfg)
		cases = append(cases, identityCase{name: fmt.Sprintf("%s/%s", img, name), img: img, w: w, h: h, sharp: sharp, exact: exact, cfg: cfg})
	}
	for m := 0; m <= 6; m++ {
		for _, tl := range []bool{false, true} {
			name := fmt.Sprintf("m%d", m)
			if tl {
				name += "-threads"
			}
			add("source.jpg", 0, 0, false, false, name, func(c *Config) { c.Method = m; c.ThreadLevel = tl })
			add("logo.jpg", 0, 0, false, false, name, func(c *Config) { c.Method = m; c.ThreadLevel = tl })
			add("synth3", 200, 136, false, false, name+"-alpha", func(c *Config) { c.Method = m; c.ThreadLevel = tl })
		}
		add("source.jpg", 0, 0, true, false, fmt.Sprintf("m%d-sharp", m), func(c *Config) { c.Method = m })
		add("synth3", 200, 136, true, true, fmt.Sprintf("m%d-sharp-exact", m), func(c *Config) { c.Method = m })
		add("synth1", 72, 40, false, false, fmt.Sprintf("m%d-noise", m), func(c *Config) { c.Method = m })
		add("synth2", 120, 88, false, false, fmt.Sprintf("m%d-edges", m), func(c *Config) { c.Method = m })
		add("synth0", 33, 17, false, false, fmt.Sprintf("m%d-gradient", m), func(c *Config) { c.Method = m })
	}
	for _, q := range []float32{0, 10, 50, 90, 100} {
		for _, m := range []int{0, 2, 4, 6} {
			add("logo.jpg", 0, 0, false, false, fmt.Sprintf("q%v-m%d", q, m), func(c *Config) { c.Quality = q; c.Method = m })
		}
	}
	add("source.jpg", 0, 0, false, false, "target-size", func(c *Config) { c.TargetSize = 20000; c.Pass = 4 })
	add("source.jpg", 0, 0, false, false, "target-psnr", func(c *Config) { c.TargetPSNR = 38; c.Pass = 4; c.Method = 6 })
	add("logo.jpg", 0, 0, false, false, "seg1-sns0", func(c *Config) { c.Segments = 1; c.SNSStrength = 0 })
	add("logo.jpg", 0, 0, false, false, "sns100-sharp7", func(c *Config) { c.SNSStrength = 100; c.FilterSharpness = 7; c.Method = 5 })
	add("logo.jpg", 0, 0, false, false, "simple-filter", func(c *Config) { c.FilterType = 0; c.Autofilter = true; c.Method = 6 })
	add("source.jpg", 0, 0, false, false, "jpeg-size", func(c *Config) { c.EmulateJPEGSize = true; c.Preprocessing = 3; c.Method = 3 })
	add("source.jpg", 0, 0, false, false, "partitions", func(c *Config) { c.Partitions = 3; c.LowMemory = true; c.PartitionLimit = 70 })
	return cases
}

func identityImage(tb testing.TB, c identityCase, cache map[string]*image.NRGBA) *image.NRGBA {
	key := fmt.Sprintf("%s/%dx%d", c.img, c.w, c.h)
	if img, ok := cache[key]; ok {
		return img
	}
	var img *image.NRGBA
	if strings.HasPrefix(c.img, "synth") {
		img = synthImage(c.w, c.h, int(c.img[5]-'0'), 7)
	} else {
		img = loadTestImage(tb, c.img)
		if c.w > 0 {
			crop := image.NewNRGBA(image.Rect(0, 0, c.w, c.h))
			draw.Draw(crop, crop.Rect, img, img.Rect.Min, draw.Src)
			img = crop
		}
	}
	cache[key] = img
	return img
}

func encodeIdentityCase(tb testing.TB, c identityCase, cache map[string]*image.NRGBA) []byte {
	tb.Helper()
	pic := FromNRGBA(identityImage(tb, c, cache), c.sharp, c.exact)
	defer pic.Release()
	cfg := c.cfg
	out, err := Encode(pic, &cfg)
	if err != nil {
		tb.Fatalf("%s: %v", c.name, err)
	}
	return out
}

func shortHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:8])
}

func TestEncodeIdentity(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	cache := map[string]*image.NRGBA{}
	got := map[string]string{}
	for _, c := range identityCases() {
		got[c.name] = shortHash(encodeIdentityCase(t, c, cache))
	}
	if os.Getenv("VP8ENC_PRINT_HASHES") != "" {
		names := make([]string, 0, len(got))
		for n := range got {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			fmt.Printf("\t%q: %q,\n", n, got[n])
		}
	}
	for name, h := range got {
		want, ok := identityHashes[name]
		if !ok {
			t.Errorf("%s: no reference hash (got %s)", name, h)
			continue
		}
		if h != want {
			t.Errorf("%s: output hash %s, want %s", name, h, want)
		}
	}
}

var identityHashes = map[string]string{
	"logo.jpg/m0":             "fcc19ac536d2d161",
	"logo.jpg/m0-threads":     "fcc19ac536d2d161",
	"logo.jpg/m1":             "76dc7f4d8cefea82",
	"logo.jpg/m1-threads":     "76dc7f4d8cefea82",
	"logo.jpg/m2":             "888a6a347c657fd0",
	"logo.jpg/m2-threads":     "888a6a347c657fd0",
	"logo.jpg/m3":             "7f09ea3e1c2eb3ff",
	"logo.jpg/m3-threads":     "7f09ea3e1c2eb3ff",
	"logo.jpg/m4":             "5d4d18e1d735cfcc",
	"logo.jpg/m4-threads":     "5d4d18e1d735cfcc",
	"logo.jpg/m5":             "1a03416c25915515",
	"logo.jpg/m5-threads":     "1a03416c25915515",
	"logo.jpg/m6":             "6e73b1e1f8d7da1d",
	"logo.jpg/m6-threads":     "6e73b1e1f8d7da1d",
	"logo.jpg/q0-m0":          "54be3a1f1699bd55",
	"logo.jpg/q0-m2":          "41c8e668287d05e2",
	"logo.jpg/q0-m4":          "9f576635fb26ac11",
	"logo.jpg/q0-m6":          "cc73841773247101",
	"logo.jpg/q10-m0":         "53d3e7ce706043f5",
	"logo.jpg/q10-m2":         "2a3521a67495f610",
	"logo.jpg/q10-m4":         "ad4932451346fdcc",
	"logo.jpg/q10-m6":         "37202ae7e08bd7bf",
	"logo.jpg/q100-m0":        "84986baf2386e151",
	"logo.jpg/q100-m2":        "ecd40b5e983b1b96",
	"logo.jpg/q100-m4":        "d5f3c4a14eee3a74",
	"logo.jpg/q100-m6":        "edc61fb4324bf0ae",
	"logo.jpg/q50-m0":         "a719f452babe5aa3",
	"logo.jpg/q50-m2":         "8486c7e85596a46d",
	"logo.jpg/q50-m4":         "661cb6d7350f627d",
	"logo.jpg/q50-m6":         "c0f1454297c5e2aa",
	"logo.jpg/q90-m0":         "2f67f2e2df3eac65",
	"logo.jpg/q90-m2":         "235e13287f0b7929",
	"logo.jpg/q90-m4":         "336c5c966c226686",
	"logo.jpg/q90-m6":         "4ca83205bcdcf72e",
	"logo.jpg/seg1-sns0":      "535bdd3bc83861ac",
	"logo.jpg/simple-filter":  "ac638b2cc10b06b1",
	"logo.jpg/sns100-sharp7":  "539d82eb85e478d1",
	"source.jpg/jpeg-size":    "3ddea2e6ed3c720c",
	"source.jpg/m0":           "356c1fa6d3deba6d",
	"source.jpg/m0-sharp":     "d01268492b631389",
	"source.jpg/m0-threads":   "356c1fa6d3deba6d",
	"source.jpg/m1":           "1ea84b9d3e57fb96",
	"source.jpg/m1-sharp":     "d47cef9b6ba452bc",
	"source.jpg/m1-threads":   "1ea84b9d3e57fb96",
	"source.jpg/m2":           "1c07f08b62778ae5",
	"source.jpg/m2-sharp":     "c2d2e29bdd0449ec",
	"source.jpg/m2-threads":   "1c07f08b62778ae5",
	"source.jpg/m3":           "8a3968048da0f4c4",
	"source.jpg/m3-sharp":     "381ff68d4f685be1",
	"source.jpg/m3-threads":   "8a3968048da0f4c4",
	"source.jpg/m4":           "9ed3379c6b0ab43a",
	"source.jpg/m4-sharp":     "d43f70cf746415b8",
	"source.jpg/m4-threads":   "9ed3379c6b0ab43a",
	"source.jpg/m5":           "d89b174bc05d1d99",
	"source.jpg/m5-sharp":     "860e91120751b367",
	"source.jpg/m5-threads":   "d89b174bc05d1d99",
	"source.jpg/m6":           "ff6112478aa903a3",
	"source.jpg/m6-sharp":     "248e73d0b9c55435",
	"source.jpg/m6-threads":   "ff6112478aa903a3",
	"source.jpg/partitions":   "05752669830ff6f0",
	"source.jpg/target-psnr":  "ba0bc7e37a3bd0f7",
	"source.jpg/target-size":  "37a8d2e8bd2a9a29",
	"synth0/m0-gradient":      "baa8a2487d77318a",
	"synth0/m1-gradient":      "a0a7371d89babe28",
	"synth0/m2-gradient":      "d80ddcd711a4a77e",
	"synth0/m3-gradient":      "7ef3e60e59d70504",
	"synth0/m4-gradient":      "7ef3e60e59d70504",
	"synth0/m5-gradient":      "00db47c9271bdfa5",
	"synth0/m6-gradient":      "c8bc5c7c20b721fa",
	"synth1/m0-noise":         "ae305f240784e607",
	"synth1/m1-noise":         "98a30a4abfcd2a14",
	"synth1/m2-noise":         "7a0f51e0bab0fa40",
	"synth1/m3-noise":         "0a9ac11c6159c7d9",
	"synth1/m4-noise":         "7f26fb4c3bcbbed3",
	"synth1/m5-noise":         "1edc03c5fe9bfa6a",
	"synth1/m6-noise":         "396a9bec290273ad",
	"synth2/m0-edges":         "17a275f767fd54f2",
	"synth2/m1-edges":         "6de8def37df0022a",
	"synth2/m2-edges":         "9e1abb8b811c28ed",
	"synth2/m3-edges":         "d21fd439b02d2bc9",
	"synth2/m4-edges":         "efeacff7ca79a3cf",
	"synth2/m5-edges":         "01413f76f7754997",
	"synth2/m6-edges":         "04977d032a604d70",
	"synth3/m0-alpha":         "46b69bbccbf075aa",
	"synth3/m0-sharp-exact":   "25600f1fbe85b283",
	"synth3/m0-threads-alpha": "46b69bbccbf075aa",
	"synth3/m1-alpha":         "2007a49ecbaed968",
	"synth3/m1-sharp-exact":   "d352ca284e4f1d8c",
	"synth3/m1-threads-alpha": "2007a49ecbaed968",
	"synth3/m2-alpha":         "c785320896b1d358",
	"synth3/m2-sharp-exact":   "7db5cee86d7979c7",
	"synth3/m2-threads-alpha": "c785320896b1d358",
	"synth3/m3-alpha":         "e1d2db3a6b1de9db",
	"synth3/m3-sharp-exact":   "455ac91e1b41589f",
	"synth3/m3-threads-alpha": "e1d2db3a6b1de9db",
	"synth3/m4-alpha":         "b6e3c4e8ee9eb53a",
	"synth3/m4-sharp-exact":   "f18462a0cbceae4f",
	"synth3/m4-threads-alpha": "b6e3c4e8ee9eb53a",
	"synth3/m5-alpha":         "e9df4e48e8bc8690",
	"synth3/m5-sharp-exact":   "d7b5609616b233b6",
	"synth3/m5-threads-alpha": "e9df4e48e8bc8690",
	"synth3/m6-alpha":         "ebec43def92c3e49",
	"synth3/m6-sharp-exact":   "7c1c20950545dde1",
	"synth3/m6-threads-alpha": "ebec43def92c3e49",
}
