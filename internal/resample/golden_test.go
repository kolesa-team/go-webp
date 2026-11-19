// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package resample

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io"
	"os"
	"testing"
)

// Golden records produced by libwebp 1.6.0 (WebPRescalerInit/Import/Export
// and WebPMultARGBRow) on random inputs. Format: op u32, params [6]u32,
// len u32, input, len u32, output.
const (
	opRescale = 6 // params: srcW, srcH, dstW, dstH, channels
	opMult    = 7 // params: inverse
)

type goldenRec struct {
	op  uint32
	p   [6]uint32
	in  []byte
	out []byte
}

func readGolden(t testing.TB, name string) []goldenRec {
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

// rescaleAll runs a full rescale of src with the streaming API.
func rescaleAll(t testing.TB, src []byte, sw, sh, dw, dh, ch int) []byte {
	rs, err := New(sw, sh, dw, dh, ch)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, dw*dh*ch)
	imported := 0
	for imported < sh {
		imported += rs.Import(sh-imported, src[imported*sw*ch:], sw*ch)
		rs.Export(out[rs.DstY()*dw*ch:], dw*ch)
	}
	if !rs.OutputDone() || !rs.InputDone() {
		t.Fatalf("rescaler not done: src %d/%d dst %d/%d", rs.SrcY(), sh, rs.DstY(), dh)
	}
	return out
}

func runOp(t testing.TB, r goldenRec) []byte {
	switch r.op {
	case opRescale:
		p := r.p
		return rescaleAll(t, r.in, int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]))
	case opMult:
		d := append([]byte(nil), r.in...)
		if r.p[0] != 0 {
			UnpremultiplyRGBA(d)
		} else {
			PremultiplyRGBA(d)
		}
		return d
	}
	return nil
}

func TestGoldenLibwebp(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		rs := readGolden(t, "testdata/golden.bin.gz")
		if len(rs) == 0 {
			t.Fatal("no golden records")
		}
		for i, r := range rs {
			if got := runOp(t, r); !bytes.Equal(got, r.out) {
				t.Errorf("record %d (op %d, params %v): mismatch with libwebp", i, r.op, r.p)
			}
		}
	})
}
