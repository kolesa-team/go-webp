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
	"math/rand"
	"testing"
)

// rescaleLegacy is rescaleAll with the legacy oracle.
func rescaleLegacy(t testing.TB, src []byte, sw, sh, dw, dh, ch int) []byte {
	var rs legacyRescaler
	if err := rs.Init(sw, sh, dw, dh, ch, nil); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, dw*dh*ch)
	imported := 0
	for imported < sh {
		imported += rs.Import(sh-imported, src[imported*sw*ch:], sw*ch)
		rs.Export(out[rs.DstY()*dw*ch:], dw*ch)
	}
	return out
}

// randomPixels returns n bytes biased towards the extremes, so that sums
// and products reach their limits.
func randomPixels(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	switch r.Intn(4) {
	case 0:
		for i := range b {
			b[i] = 255
		}
	case 1:
		for i := range b {
			b[i] = byte(255 - r.Intn(3))
		}
	default:
		r.Read(b)
	}
	return b
}

// randomGeometry returns source and destination sizes covering shrink,
// expand, mixed directions, equal sizes, extreme ratios and widths around
// the assembly cut-offs.
func randomGeometry(r *rand.Rand) (sw, sh, dw, dh int) {
	dim := func() int {
		switch r.Intn(6) {
		case 0:
			return 1 + r.Intn(4)
		case 1:
			return 1 + r.Intn(40)
		case 2:
			return 30 + r.Intn(100)
		default:
			return 1 + r.Intn(700)
		}
	}
	sw, sh, dw, dh = dim(), dim(), dim(), dim()
	switch r.Intn(5) {
	case 0: // pure horizontal shrink by a large factor
		dw = 1 + r.Intn(max(1, sw/(1+r.Intn(40))))
	case 1:
		dw = sw
	case 2:
		dh = sh
	}
	return sw, sh, dw, dh
}

// TestLegacyEquivalence checks the table-driven rescaler (with whatever
// kernels this build selects) against the legacy line-by-line port.
func TestLegacyEquivalence(t *testing.T) {
	forEachLevel(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(7))
		n := 3000
		if testing.Short() {
			n = 600
		}
		for i := 0; i < n; i++ {
			sw, sh, dw, dh := randomGeometry(r)
			ch := 1 + r.Intn(4)
			if i%3 != 0 {
				ch = []int{1, 4}[r.Intn(2)]
			}
			src := randomPixels(r, sw*sh*ch)
			want := rescaleLegacy(t, src, sw, sh, dw, dh, ch)
			got := rescaleAll(t, src, sw, sh, dw, dh, ch)
			if !bytes.Equal(got, want) {
				t.Fatalf("%dx%d -> %dx%d, %d channels: output differs from the legacy rescaler", sw, sh, dw, dh, ch)
			}
		}
	})
}

// TestColumnTable checks the column table invariants the assembly relies
// on: shrinking sums every source pixel exactly once with at least one per
// destination pixel, expanding never reads past the source row.
func TestColumnTable(t *testing.T) {
	r := rand.New(rand.NewSource(8))
	for i := 0; i < 5000; i++ {
		sw, sh, dw, dh := randomGeometry(r)
		rs, err := New(sw, sh, dw, dh, 1)
		if err != nil {
			t.Fatal(err)
		}
		if rs.xExpand {
			for x := 0; x < dw; x++ {
				l, w := int(rs.cols[2*x]), rs.cols[2*x+1]
				if l >= sw || (sw > 1 && l+1 >= sw) || w > uint32(rs.xAdd) {
					t.Fatalf("%d -> %d: pixel %d: left %d weight %d", sw, dw, x, l, w)
				}
			}
			continue
		}
		total := 0
		for x := 0; x < dw; x++ {
			k, na := int(rs.cols[2*x]), rs.cols[2*x+1]
			if k < 1 || na >= uint32(rs.xSub) {
				t.Fatalf("%d -> %d: pixel %d: count %d -accum %d", sw, dw, x, k, na)
			}
			total += k
		}
		if total != sw {
			t.Fatalf("%d -> %d: spans cover %d source pixels", sw, dw, total)
		}
	}
}
