package vp8enc

import (
	"math/rand"
	"testing"
)

// Differential tests of the batched DSP entry points (assembly on arm64,
// the reference loops elsewhere) against the reference implementations of
// dsp_batch.go, which are built on the single-block Go code of dsp.go.

var vecTables = []struct {
	name string
	t    *pairTable
}{
	{"Y16", &tabY16}, {"UV", &tabUV}, {"I4Src", &tabI4Src}, {"I4Rec", &tabI4Rec},
}

// fillPixels fills b with a random pattern: uniform noise, extremes only,
// or a narrow range around a random level (small residuals).
func fillPixels(rng *rand.Rand, b []byte) {
	switch rng.Intn(4) {
	case 0:
		for i := range b {
			b[i] = byte(rng.Intn(256))
		}
	case 1:
		for i := range b {
			b[i] = byte(rng.Intn(2) * 255)
		}
	default:
		base := rng.Intn(256)
		spread := 1 + rng.Intn(16)
		for i := range b {
			b[i] = byte(min(max(base+rng.Intn(2*spread+1)-spread, 0), 255))
		}
	}
}

// fillCoeffs fills c with coefficients of a random magnitude class,
// including the int16 extremes and DC-only blocks.
func fillCoeffs(rng *rand.Rand, c []int16) {
	scale := []int{2, 16, 300, 2048, 2400, 16384, 32768}[rng.Intn(7)]
	dcOnly := rng.Intn(5) == 0
	for i := range c {
		switch {
		case dcOnly && i%16 != 0:
			c[i] = 0
		case rng.Intn(4) == 0:
			c[i] = 0
		default:
			c[i] = int16(rng.Intn(2*scale) - scale)
		}
		switch rng.Intn(200) {
		case 0:
			c[i] = [3]int16{-32768, 32767, -32767}[rng.Intn(3)]
		case 1:
			// the bound of the 16-bit path of iTransformPairsNEON
			c[i] = [4]int16{-2048, -2047, 2047, 2048}[rng.Intn(4)]
		}
	}
}

// TestITransformBoundary runs every sign pattern of coefficients of
// magnitude 2047 (the largest the 16-bit path of the NEON inverse DCT
// accepts, where its intermediate values are largest) and 2048 (the
// smallest it rejects), for both blocks of a pair.
func TestITransformBoundary(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	ref := make([]byte, 4*bps)
	got := make([]byte, 4*bps)
	want := make([]byte, 4*bps)
	tab := newPairTable([2]uint16{0, 0})
	var in [2][16]int16
	step := 1
	if testing.Short() {
		step = 13
	}
	for p := 0; p < 1<<16; p += step {
		for i := 0; i < 16; i++ {
			v := int16(2047)
			if p>>i&1 != 0 {
				v = -2047
			}
			in[0][i] = v
			in[1][i] = -v
		}
		if p&1 != 0 {
			in[1][p>>1&15] = [2]int16{2048, -2048}[p>>5&1]
		}
		fillPixels(rng, ref)
		iTransformPairs(ref, in[:], got, &tab)
		iTransformPairsRef(ref, in[:], want, &tab)
		if string(got) != string(want) {
			t.Fatalf("pattern %#x: in %v\ngot  %v\nwant %v", p, in, got, want)
		}
	}
}

func vecIters(t *testing.T, n int) int {
	if testing.Short() {
		return n / 10
	}
	return n
}

func TestFTransformPairsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	src := make([]byte, 16*bps)
	ref := make([]byte, 16*bps)
	var got, want [16][16]int16
	for iter := 0; iter < vecIters(t, 30000); iter++ {
		fillPixels(rng, src)
		fillPixels(rng, ref)
		for _, tc := range vecTables {
			fTransformPairs(src, ref, tc.t, got[:])
			fTransformPairsRef(src, ref, tc.t, want[:])
			if got != want {
				t.Fatalf("%s iter %d: got %v, want %v", tc.name, iter, got, want)
			}
		}
	}
}

// TestFTransformExtremes runs every block pair made of the corner values
// 0 and 255 (2^16 src/ref sign patterns per 4x4 block for the largest
// coefficients).
func TestFTransformExtremes(t *testing.T) {
	src := make([]byte, 16*bps)
	ref := make([]byte, 16*bps)
	var got, want [16][16]int16
	tab := newPairTable([2]uint16{0, 0})
	step := 1
	if testing.Short() {
		step = 7
	}
	for p := 0; p < 1<<16; p += step {
		for i := 0; i < 16; i++ {
			v := byte(0)
			if p>>i&1 != 0 {
				v = 255
			}
			src[(i>>2)*bps+i&3] = v
			ref[(i>>2)*bps+i&3] = 255 - v
			src[(i>>2)*bps+4+i&3] = 255 - v
			ref[(i>>2)*bps+4+i&3] = byte(p)
		}
		fTransformPairs(src, ref, &tab, got[:])
		fTransformPairsRef(src, ref, &tab, want[:])
		if got[0] != want[0] || got[1] != want[1] {
			t.Fatalf("pattern %#x: got %v, want %v", p, got[:2], want[:2])
		}
	}
}

func TestITransformPairsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	ref := make([]byte, 16*bps)
	got := make([]byte, 16*bps)
	want := make([]byte, 16*bps)
	var in [16][16]int16
	for iter := 0; iter < vecIters(t, 30000); iter++ {
		fillPixels(rng, ref)
		fillPixels(rng, got)
		copy(want, got)
		for b := range in {
			fillCoeffs(rng, in[b][:])
		}
		for _, tc := range vecTables {
			iTransformPairs(ref, in[:], got, tc.t)
			iTransformPairsRef(ref, in[:], want, tc.t)
			if string(got) != string(want) {
				t.Fatalf("%s iter %d: in %v\ngot  %v\nwant %v", tc.name, iter, in, got, want)
			}
		}
	}
}

// testMatrices returns the quantizer matrices of every quality index and
// coefficient type, plus some random ones (with vecOK both set and not).
func testMatrices(rng *rand.Rand) []matrix {
	var ms []matrix
	for q := 0; q < 128; q++ {
		for typ := 0; typ < 3; typ++ {
			var m matrix
			switch typ {
			case 0:
				m.q[0], m.q[1] = uint32(kDcTable[q]), uint32(kAcTable[q])
			case 1:
				m.q[0], m.q[1] = uint32(kDcTable[q])*2, uint32(kAcTable2[q])
			default:
				m.q[0], m.q[1] = uint32(kDcTable[min(q, 117)]), uint32(kAcTable[q])
			}
			expandMatrix(&m, typ)
			ms = append(ms, m)
		}
	}
	for i := 0; i < 64; i++ {
		var m matrix
		for j := range m.q {
			m.q[j] = uint32(1 + rng.Intn(1<<uint(1+rng.Intn(17))))
			m.iq[j] = uint32(rng.Intn(1 << uint(1+rng.Intn(18))))
			m.bias[j] = rng.Uint32() >> uint(rng.Intn(32))
			m.sharpen[j] = uint32(rng.Intn(1 << uint(1+rng.Intn(16))))
		}
		m.prepareVec()
		ms = append(ms, m)
	}
	return ms
}

func TestQuantizeBlocksExact(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	ms := testMatrices(rng)
	var in, inRef, out, outRef [16][16]int16
	for iter := 0; iter < vecIters(t, 20000); iter++ {
		m := &ms[rng.Intn(len(ms))]
		n := 1 + rng.Intn(16)
		for b := 0; b < n; b++ {
			fillCoeffs(rng, in[b][:])
		}
		inRef = in
		for b := range out {
			for i := range out[b] {
				out[b][i] = int16(rng.Intn(65536))
			}
		}
		outRef = out
		var info, infoRef [16]uint16
		info[n-1] = 0xdead // must be overwritten
		var infoArg []uint16
		if iter%4 != 0 {
			infoArg = info[:]
		}
		nz := quantizeBlocks(in[:n], out[:], m, infoArg)
		nzRef := quantizeBlocksRef(inRef[:n], outRef[:], m, infoRef[:])
		if nz != nzRef || in != inRef || out != outRef {
			t.Fatalf("iter %d (n %d, matrix %+v): nz %#x/%#x\nin  %v\nwant %v\nout %v\nwant %v",
				iter, n, *m, nz, nzRef, in, inRef, out, outRef)
		}
		if infoArg != nil {
			if info != infoRef {
				t.Fatalf("iter %d: info %v, want %v", iter, info, infoRef)
			}
		} else if info[n-1] != 0xdead {
			t.Fatalf("iter %d: info written although nil", iter)
		}
	}
}

// isFlat is libwebp's IsFlat, the reference for flatInfo: true if the
// number of non-zero AC levels is <= thresh.
func isFlat(levels []([16]int16), thresh int) bool {
	score := 0
	for b := range levels {
		l := &levels[b]
		for i := 1; i < 16; i++ {
			v := uint16(l[i])
			score += int((v | -v) >> 15)
		}
		if score > thresh {
			return false
		}
	}
	return true
}

// TestLevelInfo checks levelInfoOf against lastNonZero and isFlat.
func TestLevelInfo(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for iter := 0; iter < 100000; iter++ {
		var c [16]int16
		for i := range c {
			if rng.Intn(3) == 0 {
				c[i] = int16(rng.Intn(65536))
			}
		}
		if iter&1 == 0 {
			c[rng.Intn(16)] = -32768
		}
		info := levelInfoOf(&c)
		if got, want := infoLast(info), lastNonZero(&c); got != want {
			t.Fatalf("%v: last %d, want %d", c, got, want)
		}
		for thresh := 0; thresh <= 15; thresh++ {
			if got, want := flatInfo([]uint16{info}, thresh), isFlat([][16]int16{c}, thresh); got != want {
				t.Fatalf("%v: flat(%d) %v, want %v", c, thresh, got, want)
			}
		}
	}
}

func TestTTransformAndSSEPairsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	a := make([]byte, 16*bps)
	b := make([]byte, 16*bps)
	var got, want [16]int32
	for iter := 0; iter < vecIters(t, 30000); iter++ {
		fillPixels(rng, a)
		fillPixels(rng, b)
		for _, tc := range vecTables {
			tTransformPairs(a, tc.t, got[:])
			tTransformPairsRef(a, tc.t, want[:])
			if got != want {
				t.Fatalf("tTransform %s iter %d: got %v, want %v", tc.name, iter, got, want)
			}
			sse4x4Pairs(a, b, tc.t, got[:])
			sse4x4PairsRef(a, b, tc.t, want[:])
			if got != want {
				t.Fatalf("sse4x4 %s iter %d: got %v, want %v", tc.name, iter, got, want)
			}
		}
		if g, w := sse16x16(a, b), sse16x16Ref(a, b); g != w {
			t.Fatalf("sse16x16 iter %d: got %d, want %d", iter, g, w)
		}
		if g, w := sse16x8(a[bps:], b[3*bps:]), sse16x8Ref(a[bps:], b[3*bps:]); g != w {
			t.Fatalf("sse16x8 iter %d: got %d, want %d", iter, g, w)
		}
	}
}

// TestVecBounds checks that the wrappers reject buffers that are one byte
// (or one block) too short instead of letting the kernels overrun them.
func TestVecBounds(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: no panic on a short buffer", name)
			}
		}()
		f()
	}
	for _, tc := range vecTables {
		tab := tc.t
		long0 := make([]byte, tab.need[0])
		long1 := make([]byte, tab.need[1])
		short0 := long0[:tab.need[0]-1]
		short1 := long1[:tab.need[1]-1]
		blocks := make([][16]int16, 2*tab.n)
		res := make([]int32, 2*tab.n)
		// exact sizes work
		fTransformPairs(long0, long1, tab, blocks)
		iTransformPairs(long0, blocks, long1, tab)
		tTransformPairs(long0, tab, res)
		sse4x4Pairs(long0, long1, tab, res)
		mustPanic(tc.name+" fTransform src", func() { fTransformPairs(short0, long1, tab, blocks) })
		mustPanic(tc.name+" fTransform ref", func() { fTransformPairs(long0, short1, tab, blocks) })
		mustPanic(tc.name+" fTransform out", func() { fTransformPairs(long0, long1, tab, blocks[:len(blocks)-1]) })
		mustPanic(tc.name+" iTransform ref", func() { iTransformPairs(short0, blocks, long1, tab) })
		mustPanic(tc.name+" iTransform dst", func() { iTransformPairs(long0, blocks, short1, tab) })
		mustPanic(tc.name+" iTransform in", func() { iTransformPairs(long0, blocks[:len(blocks)-1], long1, tab) })
		mustPanic(tc.name+" tTransform", func() { tTransformPairs(short0, tab, res) })
		mustPanic(tc.name+" tTransform out", func() { tTransformPairs(long0, tab, res[:len(res)-1]) })
		mustPanic(tc.name+" sse4x4 a", func() { sse4x4Pairs(short0, long1, tab, res) })
		mustPanic(tc.name+" sse4x4 b", func() { sse4x4Pairs(long0, short1, tab, res) })
		mustPanic(tc.name+" sse4x4 out", func() { sse4x4Pairs(long0, long1, tab, res[:len(res)-1]) })
	}
	full := make([]byte, 15*bps+16)
	mustPanic("sse16x16", func() { sse16x16(full, full[1:]) })
	mustPanic("sse16x8", func() { sse16x8(full[:7*bps+15], full) })
	var m matrix
	m.q[0], m.q[1] = 8, 8
	expandMatrix(&m, 0)
	in := make([][16]int16, 4)
	mustPanic("quantizeBlocks", func() { quantizeBlocks(in, in[:3], &m, nil) })
	mustPanic("quantizeBlocks info", func() { quantizeBlocks(in, in[:], &m, make([]uint16, 3)) })
}

// fuzzTable builds a pair table from fuzz bytes: up to 8 pairs at offsets
// inside a 16x32 work buffer.
func fuzzTable(data []byte) (pairTable, []byte) {
	if len(data) < 1 {
		return pairTable{}, data
	}
	n := int(data[0]) % 9
	data = data[1:]
	var offs [][2]uint16
	for k := 0; k < n && len(data) >= 2; k++ {
		// rows 0..12, columns 0..24
		o0 := uint16(int(data[0])%13*bps + int(data[0]>>4)%25)
		o1 := uint16(int(data[1])%13*bps + int(data[1]>>4)%25)
		offs = append(offs, [2]uint16{o0, o1})
		data = data[2:]
	}
	return newPairTable(offs...), data
}

func FuzzFTransformPairs(f *testing.F) {
	f.Add([]byte{8, 0, 0, 8, 8, 1, 2, 3, 4}, []byte{1, 2, 3}, []byte{200, 7})
	f.Add([]byte{5, 0, 0, 0, 4}, []byte{255, 0}, []byte{0, 255})
	f.Fuzz(func(t *testing.T, tab, src, ref []byte) {
		tbl, _ := fuzzTable(tab)
		if len(src) == 0 || len(ref) == 0 {
			return
		}
		a := make([]byte, tbl.need[0])
		b := make([]byte, tbl.need[1])
		for i := range a {
			a[i] = src[i%len(src)] + byte(i/len(src))
		}
		for i := range b {
			b[i] = ref[i%len(ref)] ^ byte(i/len(ref))
		}
		got := make([][16]int16, 2*tbl.n)
		want := make([][16]int16, 2*tbl.n)
		fTransformPairs(a, b, &tbl, got)
		fTransformPairsRef(a, b, &tbl, want)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("block %d: got %v, want %v", i, got[i], want[i])
			}
		}
	})
}

func FuzzITransformPairs(f *testing.F) {
	f.Add([]byte{8, 0, 0, 8, 8}, []byte{1, 2, 3}, []byte{0x80, 0, 0xff, 0x7f})
	f.Add([]byte{1, 0, 0}, []byte{255}, []byte{0, 0x80, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, tab, ref, coeffs []byte) {
		tbl, _ := fuzzTable(tab)
		if len(ref) == 0 || len(coeffs) < 2 {
			return
		}
		r := make([]byte, tbl.need[0])
		for i := range r {
			r[i] = ref[i%len(ref)] + byte(i/len(ref))
		}
		in := make([][16]int16, 2*tbl.n)
		for b := range in {
			for i := range in[b] {
				k := (b*16 + i) * 2 % (len(coeffs) - 1)
				in[b][i] = int16(uint16(coeffs[k]) | uint16(coeffs[k+1])<<8)
			}
		}
		got := make([]byte, tbl.need[1])
		want := make([]byte, tbl.need[1])
		for i := range got {
			got[i] = byte(i)
			want[i] = byte(i)
		}
		iTransformPairs(r, in, got, &tbl)
		iTransformPairsRef(r, in, want, &tbl)
		if string(got) != string(want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})
}

func FuzzQuantizeBlocks(f *testing.F) {
	f.Add(uint8(40), uint8(0), []byte{1, 2, 3, 0x80, 0, 0xff, 0x7f})
	f.Add(uint8(127), uint8(1), []byte{0xff, 0xff, 0, 0x80})
	f.Add(uint8(0), uint8(2), []byte{0x10, 0x27})
	f.Fuzz(func(t *testing.T, q, typ uint8, coeffs []byte) {
		if len(coeffs) < 2 {
			return
		}
		var m matrix
		qi := int(q) % 128
		switch typ % 3 {
		case 0:
			m.q[0], m.q[1] = uint32(kDcTable[qi]), uint32(kAcTable[qi])
		case 1:
			m.q[0], m.q[1] = uint32(kDcTable[qi])*2, uint32(kAcTable2[qi])
		default:
			m.q[0], m.q[1] = uint32(kDcTable[min(qi, 117)]), uint32(kAcTable[qi])
		}
		expandMatrix(&m, int(typ%3))
		n := 1 + len(coeffs)/32%16
		in := make([][16]int16, n)
		for b := range in {
			for i := range in[b] {
				k := (b*16 + i) * 2 % (len(coeffs) - 1)
				in[b][i] = int16(uint16(coeffs[k]) | uint16(coeffs[k+1])<<8)
			}
		}
		inRef := append([][16]int16(nil), in...)
		out := make([][16]int16, n)
		outRef := make([][16]int16, n)
		info := make([]uint16, n)
		infoRef := make([]uint16, n)
		nz := quantizeBlocks(in, out, &m, info)
		nzRef := quantizeBlocksRef(inRef, outRef, &m, infoRef)
		if nz != nzRef {
			t.Fatalf("nz %#x, want %#x", nz, nzRef)
		}
		for b := range in {
			if in[b] != inRef[b] || out[b] != outRef[b] || info[b] != infoRef[b] {
				t.Fatalf("block %d: %v %v %#x, want %v %v %#x", b, in[b], out[b], info[b], inRef[b], outRef[b], infoRef[b])
			}
		}
	})
}

func FuzzMetricPairs(f *testing.F) {
	f.Add([]byte{8, 0, 0, 8, 8, 1, 2}, []byte{1, 2, 3}, []byte{200, 7})
	f.Fuzz(func(t *testing.T, tab, pa, pb []byte) {
		tbl, _ := fuzzTable(tab)
		if len(pa) == 0 || len(pb) == 0 {
			return
		}
		a := make([]byte, max(tbl.need[0], 15*bps+16))
		b := make([]byte, max(tbl.need[1], 15*bps+16))
		for i := range a {
			a[i] = pa[i%len(pa)] + byte(i/len(pa))
		}
		for i := range b {
			b[i] = pb[i%len(pb)] ^ byte(i/len(pb))
		}
		got := make([]int32, 2*tbl.n)
		want := make([]int32, 2*tbl.n)
		tTransformPairs(a, &tbl, got)
		tTransformPairsRef(a, &tbl, want)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("tTransform %d: got %d, want %d", i, got[i], want[i])
			}
		}
		sse4x4Pairs(a, b, &tbl, got)
		sse4x4PairsRef(a, b, &tbl, want)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("sse4x4 %d: got %d, want %d", i, got[i], want[i])
			}
		}
		if g, w := sse16x16(a, b), sse16x16Ref(a, b); g != w {
			t.Fatalf("sse16x16: got %d, want %d", g, w)
		}
		if g, w := sse16x8(a, b), sse16x8Ref(a, b); g != w {
			t.Fatalf("sse16x8: got %d, want %d", g, w)
		}
	})
}

// Kernel benchmarks: the entry point (assembly on arm64) against the Go
// reference, on a whole macroblock (Y16) or the 10 intra4 modes (I4).
func BenchmarkVecKernels(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	src := make([]byte, 16*bps)
	ref := make([]byte, 16*bps)
	dst := make([]byte, 16*bps)
	// a textured source and a close prediction, as in the mode search
	for i := range src {
		src[i] = byte(60 + i%bps*4 + i/bps*2 + rng.Intn(24))
		ref[i] = byte(int(src[i]) + rng.Intn(25) - 12)
	}
	var coeffs, levels [16][16]int16
	var info [16]uint16
	var res [16]int32
	var m matrix
	m.q[0], m.q[1] = uint32(kDcTable[40]), uint32(kAcTable[40])
	expandMatrix(&m, 0)
	fTransformPairsRef(src, ref, &tabY16, coeffs[:])
	run := func(name string, f func()) {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f()
			}
		})
	}
	run("FTransformY16/vec", func() { fTransformPairs(src, ref, &tabY16, coeffs[:]) })
	run("FTransformY16/go", func() { fTransformPairsRef(src, ref, &tabY16, coeffs[:]) })
	run("FTransformI4/vec", func() { fTransformPairs(src, ref, &tabI4Src, coeffs[:]) })
	run("FTransformI4/go", func() { fTransformPairsRef(src, ref, &tabI4Src, coeffs[:]) })
	run("QuantizeY16/vec", func() {
		c := coeffs
		quantizeBlocks(c[:], levels[:], &m, info[:])
	})
	run("QuantizeY16/go", func() {
		c := coeffs
		quantizeBlocksRef(c[:], levels[:], &m, info[:])
	})
	c := coeffs
	quantizeBlocksRef(c[:], levels[:], &m, info[:])
	run("ITransformY16/vec", func() { iTransformPairs(ref, c[:], dst, &tabY16) })
	run("ITransformY16/go", func() { iTransformPairsRef(ref, c[:], dst, &tabY16) })
	big := c
	for b := range big {
		big[b][5] = 4000 // takes the int32 path
	}
	run("ITransformY16Large/vec", func() { iTransformPairs(ref, big[:], dst, &tabY16) })
	run("TTransformY16/vec", func() { tTransformPairs(src, &tabY16, res[:]) })
	run("TTransformY16/go", func() { tTransformPairsRef(src, &tabY16, res[:]) })
	run("SSE4x4I4/vec", func() { sse4x4Pairs(src, ref, &tabI4Src, res[:]) })
	run("SSE4x4I4/go", func() { sse4x4PairsRef(src, ref, &tabI4Src, res[:]) })
	run("SSE16x16/vec", func() { sse16x16(src, ref) })
	run("SSE16x16/go", func() { sse16x16Ref(src, ref) })
}

func TestHistogramExact(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	src := make([]byte, 16*bps)
	ref := make([]byte, 16*bps)
	var coeffs [16][16]int16
	for iter := 0; iter < vecIters(t, 20000); iter++ {
		fillPixels(rng, src)
		fillPixels(rng, ref)
		for _, tc := range []*pairTable{&tabY16, &tabUV} {
			m, l := collectHistogram(src, ref, tc)
			fTransformPairsRef(src, ref, tc, coeffs[:])
			mr, lr := histogramStatsRef(coeffs[:2*tc.n])
			if m != mr || l != lr {
				t.Fatalf("iter %d: got (%d, %d), want (%d, %d)", iter, m, l, mr, lr)
			}
		}
		// arbitrary coefficients, including the extremes
		n := 1 + rng.Intn(16)
		for b := 0; b < n; b++ {
			fillCoeffs(rng, coeffs[b][:])
		}
		m, l := histogramStats(coeffs[:n])
		mr, lr := histogramStatsRef(coeffs[:n])
		if m != mr || l != lr {
			t.Fatalf("iter %d: got (%d, %d), want (%d, %d) for %v", iter, m, l, mr, lr, coeffs[:n])
		}
		c0, mb := zeroBinStats(coeffs[:n])
		c0r, mbr := zeroBinStatsRef(coeffs[:n])
		if c0 != c0r || mb != mbr {
			t.Fatalf("iter %d: zeroBinStats (%d, %d), want (%d, %d) for %v", iter, c0, mb, c0r, mbr, coeffs[:n])
		}
	}
}

func FuzzZeroBinStats(f *testing.F) {
	f.Add([]byte{0, 0x80, 0xff, 0x7f, 8, 0, 7, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 2 {
			return
		}
		n := 1 + int(data[0])%16
		c := make([][16]int16, n)
		for b := range c {
			for i := range c[b] {
				k := (b*16 + i) * 2 % (len(data) - 1)
				c[b][i] = int16(uint16(data[k]) | uint16(data[k+1])<<8)
			}
		}
		c0, mb := zeroBinStats(c)
		c0r, mbr := zeroBinStatsRef(c)
		if c0 != c0r || mb != mbr {
			t.Fatalf("(%d, %d), want (%d, %d)", c0, mb, c0r, mbr)
		}
	})
}

func BenchmarkHistogram(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	src := make([]byte, 16*bps)
	ref := make([]byte, 16*bps)
	for i := range src {
		src[i] = byte(128 + rng.Intn(9) - 4)
		ref[i] = byte(128 + rng.Intn(9) - 4)
	}
	var coeffs [16][16]int16
	fTransformPairsRef(src, ref, &tabY16, coeffs[:])
	b.Run("collect", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			collectHistogram(src, ref, &tabY16)
		}
	})
	b.Run("stats", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			histogramStats(coeffs[:])
		}
	})
	b.Run("statsRef", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			histogramStatsRef(coeffs[:])
		}
	})
}

// residualCostRef is the branchy formulation of residualCost (libwebp's
// GetResidualCost_C).
func (p *encProba) residualCostRef(ctx0 int, res *residual) int {
	n := res.first
	prob := &p.coeffs[res.ctype]
	costs := &p.levelCost[res.ctype]
	p0 := prob[n][ctx0][0]
	if res.last < 0 {
		return bitCost(0, p0)
	}
	cost := 0
	if ctx0 == 0 {
		cost = bitCost(1, p0)
	}
	t := &costs[n][ctx0]
	coeffs := res.coeffs
	for ; n < res.last; n++ {
		v := abs16(coeffs[n&15])
		cost += levelCost(t, v)
		ctx := v
		if ctx > 2 {
			ctx = 2
		}
		t = &costs[encBands[n+1]][ctx]
	}
	v := abs16(coeffs[n&15])
	cost += levelCost(t, v)
	if n < 15 {
		b := encBands[n+1]
		ctx := 2
		if v == 1 {
			ctx = 1
		}
		cost += bitCost(0, prob[b][ctx][0])
	}
	return cost
}

// randomResiduals returns blocks of levels of every shape (including the
// largest levels and residuals that start after their last level) with a
// set of probabilities.
func randomResiduals(rng *rand.Rand, n int, mags []int) ([]residual, *encProba) {
	p := new(encProba)
	p.reset()
	for ty := range p.coeffs {
		for b := range p.coeffs[ty] {
			for c := range p.coeffs[ty][b] {
				for i := range p.coeffs[ty][b][c] {
					p.coeffs[ty][b][c][i] = uint8(rng.Intn(256))
				}
			}
		}
	}
	p.calculateLevelCosts()
	blocks := make([][16]int16, n)
	res := make([]residual, n)
	for i := range blocks {
		k := rng.Intn(17)
		for j := 0; j < k; j++ {
			if rng.Intn(3) > 0 {
				v := int16(1 + rng.Intn(mags[rng.Intn(len(mags))]))
				if rng.Intn(2) == 0 {
					v = -v
				}
				blocks[i][j] = v
			}
		}
		res[i].first = i & 1
		res[i].ctype = rng.Intn(numTypes)
		res[i].setCoeffs(&blocks[i])
	}
	return res, p
}

func TestResidualCostExact(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for round := 0; round < 20; round++ {
		res, p := randomResiduals(rng, 2000, []int{1, 2, 4, 66, 70, 2047})
		for i := range res {
			for ctx := 0; ctx < numCtx; ctx++ {
				if got, want := p.residualCost(ctx, &res[i]), p.residualCostRef(ctx, &res[i]); got != want {
					t.Fatalf("%+v ctx %d: cost %d, want %d", res[i], ctx, got, want)
				}
			}
		}
	}
}

func BenchmarkResidualCost(b *testing.B) {
	// mostly small levels, as in the mode search
	res, p := randomResiduals(rand.New(rand.NewSource(12)), 1024, []int{1, 1, 2, 3, 8, 100})
	b.Run("new", func(b *testing.B) {
		s := 0
		for i := 0; i < b.N; i++ {
			s += p.residualCost(i%numCtx, &res[i&1023])
		}
		_ = s
	})
	b.Run("ref", func(b *testing.B) {
		s := 0
		for i := 0; i < b.N; i++ {
			s += p.residualCostRef(i%numCtx, &res[i&1023])
		}
		_ = s
	})
}

// checkPredictors compares predLuma16 and predChroma8 with the Go
// reference on whole prediction buffers (pre-filled with the same garbage,
// so that writes outside the predictions are caught too).
func checkPredictors(t *testing.T, fill byte, yLeft, yTop, uLeft, vLeft, uvTop []byte) {
	t.Helper()
	got := make([]byte, predSize)
	want := make([]byte, predSize)
	for i := range got {
		got[i] = fill + byte(i)
		want[i] = got[i]
	}
	predLuma16(got, yLeft, yTop)
	predLuma16Ref(want, yLeft, yTop)
	predChroma8(got, uLeft, vLeft, uvTop)
	predChroma8Ref(want, uLeft, vLeft, uvTop)
	if string(got) != string(want) {
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("first difference at %d (row %d col %d): %d, want %d (left %v top %v uLeft %v vLeft %v uvTop %v)",
					i, i/bps, i%bps, got[i], want[i], yLeft, yTop, uLeft, vLeft, uvTop)
			}
		}
	}
}

func TestPredictorsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	yLeft := make([]byte, 17)
	yTop := make([]byte, 32)
	uLeft := make([]byte, 9)
	vLeft := make([]byte, 9)
	uvTop := make([]byte, 16)
	for iter := 0; iter < vecIters(t, 20000); iter++ {
		for _, b := range [][]byte{yLeft, yTop, uLeft, vLeft, uvTop} {
			fillPixels(rng, b)
		}
		checkPredictors(t, byte(iter), yLeft, yTop, uLeft, vLeft, uvTop)
		switch iter % 4 { // picture edges
		case 1:
			checkPredictors(t, byte(iter), nil, yTop, nil, nil, uvTop)
		case 2:
			checkPredictors(t, byte(iter), yLeft, nil, uLeft, vLeft, nil)
		case 3:
			checkPredictors(t, byte(iter), nil, nil, nil, nil, nil)
		}
	}
}

func FuzzPredictors(f *testing.F) {
	f.Add([]byte{1, 2, 3, 255, 0, 128})
	f.Add([]byte{255})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		b := make([]byte, 17+16+9+9+16)
		for i := range b {
			b[i] = data[i%len(data)] + byte(i/len(data))*data[0]
		}
		checkPredictors(t, data[0], b[0:17], b[17:33], b[33:42], b[42:51], b[51:67])
	})
}

// checkPredLuma4 compares predLuma4 with the Go reference on whole
// prediction buffers pre-filled with the same garbage.
func checkPredLuma4(t *testing.T, fill byte, b []byte, top int) {
	t.Helper()
	got := make([]byte, predSize)
	want := make([]byte, predSize)
	for i := range got {
		got[i] = fill ^ byte(i*7)
		want[i] = got[i]
	}
	predLuma4(got, b, top)
	predLuma4Ref(want, b, top)
	if string(got) != string(want) {
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("t=%d: first difference at %d (row %d col %d): %d, want %d (samples %v)",
					top, i, (i-i4DC4)/bps, i%bps, got[i], want[i], b[top-5:top+8])
			}
		}
	}
}

func TestPredLuma4Exact(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	var b [40]byte
	for iter := 0; iter < vecIters(t, 50000); iter++ {
		fillPixels(rng, b[:])
		top := vp8TopLeftI4[iter&15]
		if iter%5 == 0 {
			top = 5 + rng.Intn(25)
		}
		checkPredLuma4(t, byte(iter), b[:], top)
		// a boundary slice too short for the 16-byte load takes the Go code
		checkPredLuma4(t, byte(iter), b[:top+8], top)
	}
}

func FuzzPredLuma4(f *testing.F) {
	f.Add([]byte{0, 255, 128, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, uint8(0))
	f.Fuzz(func(t *testing.T, data []byte, pos uint8) {
		if len(data) == 0 {
			return
		}
		var b [40]byte
		for i := range b {
			b[i] = data[i%len(data)] ^ byte(i/len(data))
		}
		checkPredLuma4(t, byte(pos), b[:], 5+int(pos)%25)
	})
}

func BenchmarkPredLuma4(b *testing.B) {
	rng := rand.New(rand.NewSource(16))
	p := make([]byte, predSize)
	var bnd [40]byte
	fillPixels(rng, bnd[:])
	b.Run("vec", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predLuma4(p, bnd[:], 17)
		}
	})
	b.Run("go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predLuma4Ref(p, bnd[:], 17)
		}
	})
}

func BenchmarkPredictors(b *testing.B) {
	rng := rand.New(rand.NewSource(14))
	p := make([]byte, predSize)
	edge := make([]byte, 17+16+9+9+16)
	fillPixels(rng, edge)
	yLeft, yTop, uLeft, vLeft, uvTop := edge[0:17], edge[17:33], edge[33:42], edge[42:51], edge[51:67]
	b.Run("vec", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predLuma16(p, yLeft, yTop)
			predChroma8(p, uLeft, vLeft, uvTop)
		}
	})
	b.Run("go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			predLuma16Ref(p, yLeft, yTop)
			predChroma8Ref(p, uLeft, vLeft, uvTop)
		}
	})
}

// TestQuantizeBlocksLimits quantizes every int16 value in every coefficient
// position with matrices at the limits the vector quantizer accepts
// (iq = 2^16-1, sharpen = 2^15-1, so |coeff|+sharpen reaches 65535 and the
// 32-bit product plus bias wraps) and just past them (Go fallback), with
// steps whose low 16 bits wrap the dequantized value.
func TestQuantizeBlocksLimits(t *testing.T) {
	var ms []matrix
	for _, iq := range []uint32{0, 1, 0x7fff, 0xffff, 0x10000} {
		for _, sh := range []uint32{0, 1, 0x7fff, 0x8000} {
			for _, bias := range []uint32{0, 1<<17 - 1, 0xffffffff} {
				var m matrix
				for j := range m.q {
					m.q[j] = [5]uint32{1, 3, 0xffff, 0x10001, 0xfffffff}[j%5]
					m.iq[j] = iq
					m.sharpen[j] = sh
					m.bias[j] = bias - uint32(j)*0x1000
				}
				m.prepareVec()
				ms = append(ms, m)
			}
		}
	}
	step := 1
	if testing.Short() {
		step = 7
	}
	var in, inRef, out, outRef [32][16]int16
	var info, infoRef [32]uint16
	for mi := range ms {
		m := &ms[mi]
		for v := 0; v < 1<<16; v += 32 * step {
			for b := range in {
				for j := range in[b] {
					in[b][j] = int16(v + b*step + j*4099)
				}
			}
			inRef = in
			nz := quantizeBlocks(in[:], out[:], m, info[:])
			nzRef := quantizeBlocksRef(inRef[:], outRef[:], m, infoRef[:])
			if nz != nzRef || in != inRef || out != outRef || info != infoRef {
				t.Fatalf("matrix %d (iq %#x sharpen %#x bias %#x vecOK %v), v %d: results differ",
					mi, m.iq[0], m.sharpen[0], m.bias[0], m.vecOK, v)
			}
		}
	}
}
