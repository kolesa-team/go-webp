package alpha

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io"
	"os"
	"testing"
)

// Golden records produced by libwebp 1.6.0 (WebPFilters, WebPUnfilters,
// WebPEstimateBestFilter, QuantizeLevels, WebPDequantizeLevels) on random
// planes. Format: op u32, params [6]u32, len u32, input, len u32, output.
const (
	opFilter   = 1
	opUnfilter = 2
	opEstimate = 3
	opQuantize = 4
	opDequant  = 5
)

type goldenRec struct {
	op  uint32
	p   [6]uint32
	in  []byte
	out []byte
}

func readGolden(t *testing.T, name string) []goldenRec {
	t.Helper()
	f, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	var rs []goldenRec
	u32 := func() uint32 { v := binary.LittleEndian.Uint32(b); b = b[4:]; return v }
	for len(b) > 0 {
		var r goldenRec
		r.op = u32()
		for i := range r.p {
			r.p[i] = u32()
		}
		n := u32()
		r.in, b = b[:n], b[n:]
		n = u32()
		r.out, b = b[:n], b[n:]
		rs = append(rs, r)
	}
	return rs
}

// runOp applies the Go implementation of a golden record's operation.
func runOp(r goldenRec) []byte {
	w, h := int(r.p[1]), int(r.p[2])
	d := append([]byte(nil), r.in...)
	switch r.op {
	case opFilter:
		out := make([]byte, len(d))
		Filter(int(r.p[0]), d, w, h, w, out, w)
		return out
	case opUnfilter:
		Unfilter(int(r.p[0]), d, w, h, w)
		return d
	case opEstimate:
		return []byte{byte(EstimateBestFilter(d, w, h, w))}
	case opQuantize:
		if err := QuantizeLevels(d, w, h, w, int(r.p[0])); err != nil {
			return nil
		}
		return d
	case opDequant:
		if err := Dequantize(d, w, h, w, int(r.p[0])); err != nil {
			return nil
		}
		return d
	}
	return nil
}

func TestGoldenLibwebp(t *testing.T) {
	rs := readGolden(t, "testdata/golden.bin.gz")
	if len(rs) == 0 {
		t.Fatal("no golden records")
	}
	for i, r := range rs {
		if got := runOp(r); !bytes.Equal(got, r.out) {
			t.Errorf("record %d (op %d, params %v): mismatch with libwebp", i, r.op, r.p)
		}
	}
}
