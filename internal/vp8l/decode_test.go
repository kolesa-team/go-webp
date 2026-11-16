package vp8l

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"image"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/utils"
)

func must(t testing.TB, err error, ctx ...any) {
	t.Helper()
	if err != nil {
		t.Fatal(append([]any{err}, ctx...)...)
	}
}

func check(t testing.TB, ok bool, ctx ...any) {
	t.Helper()
	if !ok {
		t.Error(ctx...)
	}
}

// Expected SHA-256 of the RGBA output of dwebp -pam (libwebp 1.6.0).
var losslessFixtures = map[string]string{
	"alpha_17x33_nl60.webp":                                "a120d78c7b98c62eb70da53033bc51f1d6c311fd5aeb9d5e3916516dbbc63421",
	"alpha_17x33_z0.webp":                                  "a051284a09bb55318f310b9ce49885c835760e299459dfeddf7a92a1dc9e92a4",
	"alpha_257x129_z6.webp":                                "3a04f3ad781e46eeec07cf3ebe41529dbc1ef07071e1fa74cb95498c7fe87a32",
	"alpha_3x7_z6.webp":                                    "4fea5e6a3ec5f5474a26d858bc77b6d7bd3ab864ea02d988683fdc648602b248",
	"alpha_64x64_exact_z9.webp":                            "60531a4c53232f63540126a6c5e055519bb2b5c3b8cb7f3a03d49b04082bd136",
	"grad_1000x1_z0.webp":                                  "b31460905df56e213332dde9333fe4b34fecede15d2161eed9132294d7007f1d",
	"grad_1x1000_exact_z9.webp":                            "5e93deec658b1fb40f45ae26a857ce63036e1718b54c4363f23d1fb24967ece0",
	"grad_1x1_z6.webp":                                     "e3820096cb82366b860b8a4e668453a7aaaf423af03bdf289fa308ea03a79332",
	"grad_257x129_nl20.webp":                               "7c60b142bd9b69d189992c5a94e3821e87f12f6eb25d720d9967b0664c7db728",
	"grad_257x129_z1.webp":                                 "7c60b142bd9b69d189992c5a94e3821e87f12f6eb25d720d9967b0664c7db728",
	"noise_17x33_z6.webp":                                  "21cb1ee9d0272237ef6495af343243df3789c2d0f11f4bffdbc5202be0fdbe88",
	"noise_64x64_exact_z9.webp":                            "cf7762f6025586160e6266d1d147c1f061de0896751d8ab048ca86fd45253927",
	"noise_64x64_z0.webp":                                  "cf7762f6025586160e6266d1d147c1f061de0896751d8ab048ca86fd45253927",
	"noise_64x64_z3.webp":                                  "cf7762f6025586160e6266d1d147c1f061de0896751d8ab048ca86fd45253927",
	"pal16_500x375_z2.webp":                                "2c9fefd426938e52d1cb28d3cc345040529ee3727ef335c11eac34c943d6fe92",
	"pal16_64x64_z6.webp":                                  "d9c0e69b472e0ee533eca5cb4ec8624711f41e142dafea2a893d59cd7971eb90",
	"pal200_257x129_z8.webp":                               "2810f077611056ef33ad87047a0cca662d93a220abff08b82df00c7e4a4956bd",
	"pal200_64x64_z6.webp":                                 "bd3970627d7c2c7f863b740031b5e491cfdc667d581854ecf87c67414e2d73b5",
	"pal2_1000x1_z6.webp":                                  "7b104f6c34b6aa6e5bf1a1fb4843440b054eefde7d0d54f6a0105b0d169cc16c",
	"pal2_257x129_z6.webp":                                 "1e29ffacd73e5bb6cb3d035b9c8344ab82664ca6454da05d4e1485ac60890420",
	"pal3_1x1000_z6.webp":                                  "791133706143beb3f4f1f191370020c115c7b9aed1b8d5f88157d7ae1227efd4",
	"slide_16383x110.webp":                                 "f140a61f2dacf99ecbf63157406f110a82112b9b453a875083b9aa9800292f5a",
	"slide_2000x1000.webp":                                 "e381f467c511e13ff6cbe06e8ad25b7595a2624f07405b4d63a4e651ed209cae",
	"pal4_17x33_exact_z9.webp":                             "2dac74ecb3605a833722d95909b7d025ee07cbe26b496dda6f255f0b246cc116",
	"../../../../test_data/images/webp-logo-lossless.webp": "b0325033a6bf401450ddf6c461861f0d2133d211a114c889d7c44e35e7cd19e5",
	"../../../../test_data/images/100x150_lossless.webp":   "0946e2eb0fb9ea7ddd935efd1922bc7d1f27101c69ce6d2f5145c7ee28f1b6ba",
}

var alphaFixtures = map[string]string{
	"alpha_257x129_aq30.webp": "88af6569e8149a9fa9d467c81191f0d1cb146a4475f2cbaba1f9a03dd1211179",
	"alpha_64x64_aq10.webp":   "5ffff3db4014c2faddb3a409e3249b5831d5d9ec8936b33c8499a4a51b526cde",
	"alpha_64x64_aq30.webp":   "4fecfb9db51a6792ce95aa7139ca57126b62656dfd132595e4526722b569f265",
	"alpha_64x64_aq60f.webp":  "fdedda432bb6e088538450171ca62ca8a2ead7bd4fe506c1b09f1627ebf3a408",
	"alpha_64x64_best.webp":   "733a61a19bee1b6c9c3cd8f6dfc76519e7f8edceefb60fab2100e9c4169fc605",
	"alpha_64x64_fast.webp":   "733a61a19bee1b6c9c3cd8f6dfc76519e7f8edceefb60fab2100e9c4169fc605",
	"alpha_64x64_none.webp":   "733a61a19bee1b6c9c3cd8f6dfc76519e7f8edceefb60fab2100e9c4169fc605",
	"pal3_17x33_aq10.webp":    "c82ba1aef054cde06251832495a8c5bd0bd3d79e697e933ca166f6251bc50bbd",
	"pal4_257x129_aq10.webp":  "6554955b54c760c638b7a713e15eadd2d2bd5d85bd5feddcd0e9590b32904645",
}

func readFixture(t testing.TB, dir, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", dir, name))
	must(t, err)
	return b
}

func TestDecodeFixtures(t *testing.T) {
	for name, want := range losslessFixtures {
		t.Run(filepath.Base(name), func(t *testing.T) {
			payload := findChunk(readFixture(t, "lossless", name), "VP8L")
			check(t, payload != nil, "no VP8L chunk")
			img, err := decodeVP8L(payload)
			must(t, err)
			sum := sha256.Sum256(img.Pix)
			check(t, want == hex.EncodeToString(sum[:]), "sha256 mismatch")
		})
	}
}

func TestDecodeAlphaFixtures(t *testing.T) {
	for name, want := range alphaFixtures {
		t.Run(name, func(t *testing.T) {
			hdr, payload, w, h, ok := alphaFromFile(readFixture(t, "alpha", name))
			if !ok || hdr&3 != 1 {
				t.Fatal("bad ALPH chunk")
			}
			dst := make([]byte, w*h)
			must(t, DecodeAlpha(payload, w, h, dst))
			unfilterAlpha(dst, w, h, int(hdr>>2&3))
			sum := sha256.Sum256(dst)
			check(t, want == hex.EncodeToString(sum[:]), "sha256 mismatch")
		})
	}
}

// TestDecodeAlphaMatchesDecode checks DecodeAlpha against the green channel
// of a full decode of the same stream (the ALPH payload is a headerless
// VP8L stream).
func TestDecodeAlphaMatchesDecode(t *testing.T) {
	for _, name := range []string{"noise_64x64_z3.webp", "alpha_257x129_z6.webp", "pal4_17x33_exact_z9.webp", "pal16_500x375_z2.webp"} {
		payload := findChunk(readFixture(t, "lossless", name), "VP8L")
		img, err := decodeVP8L(payload)
		must(t, err)
		w, h := img.Rect.Dx(), img.Rect.Dy()
		dst := make([]byte, w*h)
		must(t, DecodeAlpha(payload[headerSize:], w, h, dst), name)
		for i := range dst {
			if dst[i] != img.Pix[4*i+1] {
				t.Fatalf("%s: mismatch at %d", name, i)
			}
		}
	}
}

func TestDecodeHeader(t *testing.T) {
	payload := findChunk(readFixture(t, "lossless", "alpha_257x129_z6.webp"), "VP8L")
	w, h, a, err := DecodeHeader(payload)
	must(t, err)
	check(t, w == 257 && h == 129 && a, w, h, a)

	_, _, _, err = DecodeHeader(payload[:4])
	check(t, status.CodeOf(err) == utils.Vp8StatusNotEnoughData, err)

	bad := append([]byte(nil), payload[:5]...)
	bad[0] = 0x2e
	_, _, _, err = DecodeHeader(bad)
	check(t, status.CodeOf(err) == utils.Vp8StatusBitstreamError, err)

	bad[0] = signature
	bad[4] |= 0x20 // version 1
	_, _, _, err = DecodeHeader(bad)
	check(t, status.CodeOf(err) == utils.Vp8StatusBitstreamError, err)
}

func TestDecodeTruncated(t *testing.T) {
	payload := findChunk(readFixture(t, "lossless", "pal200_64x64_z6.webp"), "VP8L")
	for n := 5; n < len(payload); n += 1 + n/16 {
		_, err := decodeVP8L(payload[:n])
		var se *status.Error
		if !errors.As(err, &se) {
			t.Fatalf("len %d: got %v, want *status.Error", n, err)
		}
	}
}

// rowSink records the rows it receives and stops after need rows.
type rowSink struct {
	img   *image.NRGBA
	need  int
	next  int
	calls int
	fail  error
}

func (s *rowSink) Buffer(y, n int) []byte {
	if y != s.next {
		panic("rows out of order")
	}
	return s.img.Pix[y*s.img.Stride : (y+n)*s.img.Stride]
}
func (s *rowSink) Stride() int { return s.img.Stride }
func (s *rowSink) Commit(y, n int) error {
	s.next = y + n
	s.calls++
	return s.fail
}
func (s *rowSink) NeedRows() int { return s.need }

func TestDecodeNeedRows(t *testing.T) {
	for _, name := range []string{"alpha_257x129_z6.webp", "pal2_257x129_z6.webp", "grad_1x1000_exact_z9.webp"} {
		payload := findChunk(readFixture(t, "lossless", name), "VP8L")
		full, err := decodeVP8L(payload)
		must(t, err)
		w, h := full.Rect.Dx(), full.Rect.Dy()
		for _, need := range []int{0, 1, 15, 16, 17, h / 2, h} {
			s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, w, h)), need: need}
			must(t, Decode(payload, s), name, need)
			check(t, s.next == min(need, h), name, need, s.next)
			n := min(need, h) * full.Stride
			check(t, bytes.Equal(full.Pix[:n], s.img.Pix[:n]), name, need)
		}
	}
}

func TestDecodeSinkError(t *testing.T) {
	payload := findChunk(readFixture(t, "lossless", "alpha_257x129_z6.webp"), "VP8L")
	boom := errors.New("boom")
	s := &rowSink{img: image.NewNRGBA(image.Rect(0, 0, 257, 129)), need: 129, fail: boom}
	check(t, errors.Is(Decode(payload, s), boom))
	check(t, s.calls == 1, s.calls)
}

func TestDecodeAlphaParams(t *testing.T) {
	check(t, status.CodeOf(DecodeAlpha(nil, 0, 1, nil)) == utils.Vp8StatusInvalidParam)
	check(t, status.CodeOf(DecodeAlpha(nil, 4, 4, make([]byte, 15))) == utils.Vp8StatusInvalidParam)
	check(t, DecodeAlpha(nil, 4, 4, make([]byte, 16)) != nil)
}

func TestDecodeConcurrent(t *testing.T) {
	var payloads [][]byte
	var want []string
	for name, sum := range losslessFixtures {
		payloads = append(payloads, findChunk(readFixture(t, "lossless", name), "VP8L"))
		want = append(want, sum)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := range payloads {
				k := (i + g) % len(payloads)
				img, err := decodeVP8L(payloads[k])
				if err != nil {
					t.Error(err)
					return
				}
				sum := sha256.Sum256(img.Pix)
				if hex.EncodeToString(sum[:]) != want[k] {
					t.Errorf("payload %d mismatch", k)
				}
			}
		}(g)
	}
	wg.Wait()
}

func TestDirectSink(t *testing.T) {
	payload := findChunk(readFixture(t, "lossless", "grad_257x129_z1.webp"), "VP8L")
	img := image.NewNRGBA(image.Rect(0, 0, 257, 129))
	must(t, Decode(payload, pipeline.NewDirect(img)))
	check(t, img.Pix[3] == 255)
}
