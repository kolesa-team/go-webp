package vp8lenc

import "testing"

// TestPoolBounds checks that buffers sized for large images are not kept in
// the pools (they used to retain hundreds of MB after a 12 MP encode).
func TestPoolBounds(t *testing.T) {
	n := maxPooledPixels + 1
	b := getU32(n)
	if cap(b) != n {
		t.Errorf("getU32(%d): cap %d, want the exact size", n, cap(b))
	}
	putU32(b) // must not be pooled
	if b2 := getU32(n); &b2[0] == &b[0] {
		t.Error("large buffer was pooled")
	}

	s := new(scratch)
	s.noteSize(n)
	s.resid = make([]uint32, 16)
	putScratch(s)
	for i := 0; i < 4; i++ {
		if getScratch() == s {
			t.Fatal("scratch of a large image was pooled")
		}
	}
}

// TestRefsSizing checks that the backward reference buffers follow the size
// of the parse: small for flat images (they used to be sized for one token
// per pixel, 96 MB for 6 MP), grown up to at most one entry per pixel for
// noise, and always covering every pixel.
func TestRefsSizing(t *testing.T) {
	cover := func(refs []pixOrCopy) int {
		n := 0
		for _, r := range refs {
			n += r.length()
		}
		return n
	}
	const w, h = 512, 512
	n := w * h
	flat := make([]uint32, n)
	for i := range flat {
		flat[i] = 0xff000000 | uint32(i/w%4)
	}
	noise := make([]uint32, n)
	seed := uint32(1)
	for i := range noise {
		seed = seed*1664525 + 1013904223
		noise[i] = seed
	}
	sc := new(scratch)
	getBackwardRefs(flat, w, h, 75, false, 0, false, 0, sc, nil)
	for i := 0; i < 2; i++ {
		if c := cap(sc.refs[i]); c > n/16 {
			t.Errorf("flat refs[%d]: cap %d, want <= %d", i, c, n/16)
		}
		if c := cover(sc.refs[i]); c != n {
			t.Errorf("flat refs[%d] covers %d pixels, want %d", i, c, n)
		}
	}
	getBackwardRefs(noise, w, h, 75, false, 0, false, 0, sc, nil)
	for i := 0; i < 2; i++ {
		if len(sc.refs[i]) < n/2 || cap(sc.refs[i]) > n {
			t.Errorf("noise refs[%d]: len %d cap %d, n %d", i, len(sc.refs[i]), cap(sc.refs[i]), n)
		}
		if c := cover(sc.refs[i]); c != n {
			t.Errorf("noise refs[%d] covers %d pixels, want %d", i, c, n)
		}
	}
}
