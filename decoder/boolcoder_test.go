package decoder

import (
	"crypto/sha256"
	"encoding/hex"
	"image"
	"os"
	"strings"
	"testing"
)

// Corrupt VP8 partitions can push the bool decoder's value more than 2^31
// above split. libwebp's VP8GetBit compares unsigned, so a signed 32-bit
// difference would decode different bits (or succeed where libwebp fails).
// Expected outputs come from a plain-C (non-NEON) libwebp 1.6 dwebp build.
func TestBoolDecoderUnsignedCompare(t *testing.T) {
	t.Run("pixels match libwebp", func(t *testing.T) {
		f, err := os.Open("testdata/bigvalue_flip.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		d, err := NewDecoder(f, nil)
		if err != nil {
			t.Fatal(err)
		}
		img, err := d.Decode()
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(img.(*image.NRGBA).Pix)
		const want = "3f080bc905aa17b0421d020389801b4f5a837b71756f0a682edc56cafb27b8f2"
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Fatalf("pixel hash %s, want %s", got, want)
		}
	})
	t.Run("truncation detected like libwebp", func(t *testing.T) {
		f, err := os.Open("testdata/bigvalue_truncated.webp")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		d, err := NewDecoder(f, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = d.Decode(); err == nil || !strings.Contains(err.Error(), "VP8_STATUS_NOT_ENOUGH_DATA") {
			t.Fatalf("got %v, want VP8_STATUS_NOT_ENOUGH_DATA", err)
		}
	})
}
