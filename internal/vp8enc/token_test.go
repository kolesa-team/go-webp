package vp8enc

import "testing"

// TestTokenBufferPages checks the paged token buffer across page boundaries:
// indices, chunked iteration and page reuse after reset.
func TestTokenBufferPages(t *testing.T) {
	var tb tokenBuffer
	defer tb.release()
	var s uint32
	for round := 0; round < 2; round++ {
		tb.reset()
		n := 2*tokenPageSize + 123
		for i := 0; i < n; i++ {
			if i%3 == 0 {
				tb.addConstant(i&1, i&0xff)
			} else {
				tb.add(i&1, i&0x3ff, &s)
			}
		}
		if tb.len() != n {
			t.Fatalf("len %d, want %d", tb.len(), n)
		}
		if len(tb.pages) != 3 {
			t.Fatalf("%d pages, want 3", len(tb.pages))
		}
		for _, r := range [][2]int{{0, n}, {tokenPageSize - 5, tokenPageSize + 5}, {7, 7}, {n - 1, n}} {
			i := r[0]
			for i < r[1] {
				c := tb.chunk(i, r[1])
				if len(c) == 0 {
					t.Fatalf("empty chunk at %d", i)
				}
				for j, v := range c {
					k := i + j
					want := uint16((k&1)<<15 | k&0x3ff)
					if k%3 == 0 {
						want = uint16((k&1)<<15 | tokenFixedProba | k&0xff)
					}
					if v != want {
						t.Fatalf("token %d = %#x, want %#x", k, v, want)
					}
				}
				i += len(c)
			}
		}
	}
}
