package vp8lenc

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"os"
	"sort"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/workers"
)

// The memory and goroutine work on the encoder must not change its output.
// identityHashes holds the SHA-256 (first 16 hex digits) of each case's
// output as produced at 88ca2a7 (before the single parse buffer, the shared
// retention budget and the per-encode helper set); TestEncodeIdentity checks
// them in every build, sequentially and with ThreadLevel under several
// helper budgets. Set VP8LENC_PRINT_HASHES=1 to print the current hashes in
// the format of the table.

// losslessLevels are libwebp's WebPConfigLosslessPreset (method, quality)
// pairs of the levels 0-9.
var losslessLevels = [10][2]int{{0, 0}, {1, 20}, {2, 25}, {3, 30}, {3, 50}, {4, 50}, {4, 75}, {4, 90}, {5, 90}, {6, 100}}

type identityCase struct {
	name  string
	img   string
	cfg   Config
	alpha int // > 0: EncodeAlpha of the image's green channel with method alpha-1
}

func identityCases() []identityCase {
	var cases []identityCase
	add := func(img, name string, cfg Config) {
		cases = append(cases, identityCase{name: img + "/" + name, img: img, cfg: cfg})
	}
	for lv, mq := range losslessLevels {
		cfg := Config{Method: mq[0], Quality: float32(mq[1]), NearLossless: 100}
		// (The brute-force level 9 is slow: small crops.)
		sfx := ""
		if lv == 9 {
			sfx = "-small"
		}
		for _, img := range []string{"photo", "logo", "cutout", "pal12", "pal200"} {
			add(img+sfx, fmt.Sprintf("z%d", lv), cfg)
		}
		c := cfg
		c.Exact = true
		add("cutout"+sfx, fmt.Sprintf("z%d-exact", lv), c)
		c = cfg
		c.LowMemory = true
		add("photo"+sfx, fmt.Sprintf("z%d-lowmem", lv), c)
		c = cfg
		c.NearLossless = 60
		add("photo"+sfx, fmt.Sprintf("z%d-nl60", lv), c)
	}
	add("photo", "nl80", Config{Method: 4, Quality: 75, NearLossless: 80})
	add("cutout", "nl40", Config{Method: 4, Quality: 75, NearLossless: 40})
	add("pal12", "graph", Config{Method: 4, Quality: 75, NearLossless: 100, ImageHint: 3})
	add("pal200", "graph", Config{Method: 5, Quality: 75, NearLossless: 100, ImageHint: 3})
	add("photo480", "z6", Config{Method: 4, Quality: 75, NearLossless: 100})
	add("photo480", "z3", Config{Method: 3, Quality: 30, NearLossless: 100})
	for m := 0; m <= 6; m++ {
		for _, q := range []float32{90, 100} {
			img := "logo"
			if q == 100 {
				if m < 6 {
					continue
				}
				img = "logo-small"
			}
			cases = append(cases, identityCase{name: fmt.Sprintf("alpha/m%d-q%v", m, q), img: img, cfg: Config{Quality: q}, alpha: m + 1})
		}
	}
	return cases
}

func identityImages(tb testing.TB) map[string]*image.NRGBA {
	tb.Helper()
	photo480 := downscale(loadTestJPEG(tb, "source.jpg"), 1)
	rng := rand.New(rand.NewSource(11))
	m := map[string]*image.NRGBA{
		"photo480": photo480,
		"photo":    downscale(photo480, 1),
		"logo":     downscale(loadTestJPEG(tb, "logo.jpg"), 2),
		"cutout": newNRGBA(173, 131, func(x, y int) color.NRGBA {
			d := (x-86)*(x-86) + (y-65)*(y-65)
			a := uint8(255)
			if d > 60*60 {
				a = 0
			} else if d > 52*52 {
				a = uint8(rng.Intn(256))
			}
			return color.NRGBA{uint8(x*3 + rng.Intn(5)), uint8(y * 2), uint8(x ^ y), a}
		}),
	}
	for _, nc := range []int{12, 200} {
		pal := make([]color.NRGBA, nc)
		for i := range pal {
			pal[i] = color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(255 - rng.Intn(2)*128)}
		}
		m[fmt.Sprintf("pal%d", nc)] = newNRGBA(211, 97, func(x, y int) color.NRGBA {
			if rng.Intn(9) == 0 {
				return pal[rng.Intn(nc)]
			}
			return pal[(x/5+y/3)%nc]
		})
	}
	for _, name := range []string{"photo", "logo", "cutout", "pal12", "pal200"} {
		m[name+"-small"] = m[name].SubImage(image.Rect(13, 11, 13+97, 11+71)).(*image.NRGBA)
	}
	return m
}

func encodeIdentityCase(tb testing.TB, c identityCase, imgs map[string]*image.NRGBA) []byte {
	tb.Helper()
	img := imgs[c.img]
	var out []byte
	var err error
	if c.alpha > 0 {
		w, h := img.Rect.Dx(), img.Rect.Dy()
		plane := make([]byte, w*h)
		for i := range plane {
			plane[i] = img.Pix[(i/w)*img.Stride+4*(i%w)+1] &^ 7
		}
		var a AlphaEncoder
		a.Interrupt = c.cfg.Interrupt
		// Two planes through one encoder (the second reuses its buffers).
		if _, err = a.Encode(plane[:w*(h/2)], w, h/2, w, c.alpha-1, c.cfg.Quality); err == nil {
			out, err = a.Encode(plane, w, h, w, c.alpha-1, c.cfg.Quality)
			out = append([]byte(nil), out...)
		}
		a.Release()
	} else {
		cfg := c.cfg
		out, err = Encode(img, &cfg)
	}
	if err != nil {
		tb.Fatalf("%s: %v", c.name, err)
	}
	return out
}

func shortHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:8])
}

func checkIdentity(t *testing.T, label string, got map[string]string) {
	t.Helper()
	for name, h := range got {
		want, ok := identityHashes[name]
		if !ok {
			t.Errorf("%s %s: no reference hash (got %s)", label, name, h)
			continue
		}
		if h != want {
			t.Errorf("%s %s: output hash %s, want %s", label, name, h, want)
		}
	}
}

// TestEncodeIdentity checks the output of every level, with and without
// exact, near-lossless, LowMemory, the graph hint and the alpha encoder,
// against the hashes of 88ca2a7, sequentially and with ThreadLevel under
// helper budgets of 1 to 8 (the parallel stages forced on these small
// images) and with the budget exhausted.
func TestEncodeIdentity(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	imgs := identityImages(t)
	cases := identityCases()
	got := map[string]string{}
	for _, c := range cases {
		got[c.name] = shortHash(encodeIdentityCase(t, c, imgs))
	}
	if os.Getenv("VP8LENC_PRINT_HASHES") != "" {
		names := make([]string, 0, len(got))
		for n := range got {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			fmt.Printf("\t%q: %q,\n", n, got[n])
		}
	}
	checkIdentity(t, "sequential", got)

	defer forceParallel()()
	for _, b := range []struct {
		limit int
		taken bool
	}{{1, false}, {3, false}, {8, false}, {4, true}} {
		t.Run(fmt.Sprintf("threads-limit%d-taken%v", b.limit, b.taken), func(t *testing.T) {
			defer workers.SetLimitForTest(workers.SetLimitForTest(b.limit))
			if b.taken {
				defer takeBudget(t)()
			}
			got := map[string]string{}
			for _, c := range cases {
				if c.alpha > 0 {
					continue // (no ThreadLevel)
				}
				c.cfg.ThreadLevel = true
				got[c.name] = shortHash(encodeIdentityCase(t, c, imgs))
			}
			checkIdentity(t, "ThreadLevel", got)
		})
	}
}

var identityHashes = map[string]string{
	"alpha/m0-q90":          "b45ae86a405160b8",
	"alpha/m1-q90":          "562ca0d72db801cb",
	"alpha/m2-q90":          "74ec4ddf0f694859",
	"alpha/m3-q90":          "da35bceb0db0a4b8",
	"alpha/m4-q90":          "643d7fa3e44e4c0e",
	"alpha/m5-q90":          "c579a884ec8a7490",
	"alpha/m6-q100":         "df6d775979a08e48",
	"alpha/m6-q90":          "977877056711167a",
	"cutout-small/z9":       "6d87a96fb6cbf5cb",
	"cutout-small/z9-exact": "930352c88dfb99a7",
	"cutout/nl40":           "7fb83fca0c68e664",
	"cutout/z0":             "7839a569804e9de1",
	"cutout/z0-exact":       "17b71739f46f38c5",
	"cutout/z1":             "058de5897282b446",
	"cutout/z1-exact":       "348b3a0d8bc0cdfb",
	"cutout/z2":             "66fe4c1c14651846",
	"cutout/z2-exact":       "34f39bb9401f48f4",
	"cutout/z3":             "c745705f2f3c8e75",
	"cutout/z3-exact":       "a878d2edf09247e1",
	"cutout/z4":             "0a6d8ce1138a8533",
	"cutout/z4-exact":       "17e8d030f282d0b2",
	"cutout/z5":             "a5fb5966dffffdd0",
	"cutout/z5-exact":       "04b8ce927d338597",
	"cutout/z6":             "7fd8ae5ac0d19dd6",
	"cutout/z6-exact":       "8c9f27b3739198f7",
	"cutout/z7":             "7fd8ae5ac0d19dd6",
	"cutout/z7-exact":       "73e3319d0be6eddc",
	"cutout/z8":             "439a37a245b0ab6a",
	"cutout/z8-exact":       "a5e8ffd56e325f2f",
	"logo-small/z9":         "5cf920bcf3d13c17",
	"logo/z0":               "f401db57d9986998",
	"logo/z1":               "6d28f5c61a906e8b",
	"logo/z2":               "80a8a296090a1b7e",
	"logo/z3":               "d7d47d0ad22ac063",
	"logo/z4":               "424caabf27c9503a",
	"logo/z5":               "46518c6d7821bb82",
	"logo/z6":               "b7857981fbcdcb3f",
	"logo/z7":               "b7857981fbcdcb3f",
	"logo/z8":               "40e17a998d0ef215",
	"pal12-small/z9":        "d53c8adfebdd5ade",
	"pal12/graph":           "6830bbb81c4bc5fa",
	"pal12/z0":              "4d3c9863819be1fa",
	"pal12/z1":              "28591f24f3526d05",
	"pal12/z2":              "c1a3d6319eb2e904",
	"pal12/z3":              "3815c060ddd64bc3",
	"pal12/z4":              "f6550f687605b512",
	"pal12/z5":              "f6550f687605b512",
	"pal12/z6":              "6830bbb81c4bc5fa",
	"pal12/z7":              "e300cd71b94b41ac",
	"pal12/z8":              "e300cd71b94b41ac",
	"pal200-small/z9":       "624c2d73b63f4386",
	"pal200/graph":          "145f419de410c77a",
	"pal200/z0":             "6c8616f0931bba63",
	"pal200/z1":             "526132f90aea0df6",
	"pal200/z2":             "458f33c95d9bcae4",
	"pal200/z3":             "2862af519884d3f0",
	"pal200/z4":             "2f3529de8f27fcec",
	"pal200/z5":             "2f3529de8f27fcec",
	"pal200/z6":             "145f419de410c77a",
	"pal200/z7":             "145f419de410c77a",
	"pal200/z8":             "145f419de410c77a",
	"photo-small/z9":        "7d33f1acafa0b7f4",
	"photo-small/z9-lowmem": "7d33f1acafa0b7f4",
	"photo-small/z9-nl60":   "300c556772a8d13d",
	"photo/nl80":            "510b2593b7295e3b",
	"photo/z0":              "79c173ee52d94052",
	"photo/z0-lowmem":       "79c173ee52d94052",
	"photo/z0-nl60":         "79c173ee52d94052",
	"photo/z1":              "ced1168363b5760a",
	"photo/z1-lowmem":       "ced1168363b5760a",
	"photo/z1-nl60":         "eea6fc9e826fd910",
	"photo/z2":              "a3453c5a374b4d5e",
	"photo/z2-lowmem":       "a3453c5a374b4d5e",
	"photo/z2-nl60":         "8e2a9dd03ff30772",
	"photo/z3":              "aa2995858a8a4ecb",
	"photo/z3-lowmem":       "aa2995858a8a4ecb",
	"photo/z3-nl60":         "fde270cecff8af06",
	"photo/z4":              "2be31e8330f5a1a1",
	"photo/z4-lowmem":       "2be31e8330f5a1a1",
	"photo/z4-nl60":         "c738858bfb49e53e",
	"photo/z5":              "a63e8aea16ebe877",
	"photo/z5-lowmem":       "a63e8aea16ebe877",
	"photo/z5-nl60":         "92834b745106356b",
	"photo/z6":              "762a404a79682493",
	"photo/z6-lowmem":       "762a404a79682493",
	"photo/z6-nl60":         "e04c4247b176ccf9",
	"photo/z7":              "d6d10a8029f86e6a",
	"photo/z7-lowmem":       "d6d10a8029f86e6a",
	"photo/z7-nl60":         "cb0506ada7c20aba",
	"photo/z8":              "621199b308cca990",
	"photo/z8-lowmem":       "621199b308cca990",
	"photo/z8-nl60":         "e71b7d0d7c706587",
	"photo480/z3":           "a224ea8b64f5f5b6",
	"photo480/z6":           "471d43a1804dc0b0",
}
