package vp8

// Algorithm adapted from libwebp (BSD-3-Clause): dsp/dec.c (simple and
// complex in-loop filters).
//
// Each filter comes in two flavours: "V" filters a horizontal edge (the
// pixels across the edge are one stride apart, the loop walks along a row)
// and "H" filters a vertical edge (pixels across the edge are adjacent, the
// loop walks down a column).

func abs(x int) int { return max(x, -x) }

func sclip1(x int) int { return min(max(x, -128), 127) }
func sclip2(x int) int { return min(max(x, -16), 15) }
func clip1(x int) uint8 {
	return uint8(min(max(x, 0), 255))
}

// filterTables are lookup tables replacing clamps in the filter kernels.
type filterTables struct {
	clip1  [1024]uint8 // clip1[(x+256)&1023] = clamp(x, 0, 255), x in [-256, 767]
	sclip2 [256]int8   // sclip2[uint8(x)] = clamp(x, -16, 15), x in [-128, 127]
}

// fTab is a pointer variable so that hot loops keep it in a register.
var fTab = func() *filterTables {
	t := new(filterTables)
	for i := range t.clip1 {
		t.clip1[i] = clip1(i - 256)
	}
	for i := range t.sclip2 {
		t.sclip2[i] = int8(sclip2(int(int8(i))))
	}
	return t
}()

// --- simple filter ----------------------------------------------------------

func simpleV(buf []byte, o, stride, n, thresh2 int) {
	t := fTab
	r1 := buf[o-2*stride:][:n]
	r2 := buf[o-stride:][:n]
	r3 := buf[o:][:n]
	r4 := buf[o+stride:][:n]
	for i := range r1 {
		p1, p0, q0, q1 := int(r1[i]), int(r2[i]), int(r3[i]), int(r4[i])
		if 4*abs(p0-q0)+abs(p1-q1) > thresh2 {
			continue
		}
		a := 3*(q0-p0) + sclip1(p1-q1)
		a1 := int(t.sclip2[uint8((a+4)>>3)])
		a2 := int(t.sclip2[uint8((a+3)>>3)])
		r2[i] = t.clip1[(p0+a2+256)&1023]
		r3[i] = t.clip1[(q0-a1+256)&1023]
	}
}

func simpleH(buf []byte, o, stride, n, thresh2 int) {
	t := fTab
	row := buf[o-2:]
	for j := 0; j < n; j++ {
		if j > 0 {
			row = row[stride:]
		}
		w := (*[4]byte)(row)
		p1, p0, q0, q1 := int(w[0]), int(w[1]), int(w[2]), int(w[3])
		if 4*abs(p0-q0)+abs(p1-q1) > thresh2 {
			continue
		}
		a := 3*(q0-p0) + sclip1(p1-q1)
		a1 := int(t.sclip2[uint8((a+4)>>3)])
		a2 := int(t.sclip2[uint8((a+3)>>3)])
		w[1] = t.clip1[(p0+a2+256)&1023]
		w[2] = t.clip1[(q0-a1+256)&1023]
	}
}

// --- complex filter -----------------------------------------------------------

// loop26V filters a macroblock edge (up to 3 pixels modified on each side).
func loop26V(buf []byte, o, stride, n, thresh2, ithresh, hevT int) {
	t := fTab
	r0 := buf[o-4*stride:][:n]
	r1 := buf[o-3*stride:][:n]
	r2 := buf[o-2*stride:][:n]
	r3 := buf[o-stride:][:n]
	r4 := buf[o:][:n]
	r5 := buf[o+stride:][:n]
	r6 := buf[o+2*stride:][:n]
	r7 := buf[o+3*stride:][:n]
	for i := range r0 {
		p3, p2, p1, p0 := int(r0[i]), int(r1[i]), int(r2[i]), int(r3[i])
		q0, q1, q2, q3 := int(r4[i]), int(r5[i]), int(r6[i]), int(r7[i])
		if p0 == q0 && p1 == q1 {
			continue // the filter would leave this line unchanged
		}
		dp, dq := abs(p1-p0), abs(q1-q0)
		if 4*abs(p0-q0)+abs(p1-q1) > thresh2 ||
			max(abs(p3-p2), abs(p2-p1), dp, abs(q3-q2), abs(q2-q1), dq) > ithresh {
			continue
		}
		if max(dp, dq) > hevT {
			a := 3*(q0-p0) + sclip1(p1-q1)
			a1 := int(t.sclip2[uint8((a+4)>>3)])
			a2 := int(t.sclip2[uint8((a+3)>>3)])
			r3[i] = t.clip1[(p0+a2+256)&1023]
			r4[i] = t.clip1[(q0-a1+256)&1023]
		} else {
			a := sclip1(3*(q0-p0) + sclip1(p1-q1))
			a1 := (27*a + 63) >> 7
			a2 := (18*a + 63) >> 7
			a3 := (9*a + 63) >> 7
			r1[i] = t.clip1[(p2+a3+256)&1023]
			r2[i] = t.clip1[(p1+a2+256)&1023]
			r3[i] = t.clip1[(p0+a1+256)&1023]
			r4[i] = t.clip1[(q0-a1+256)&1023]
			r5[i] = t.clip1[(q1-a2+256)&1023]
			r6[i] = t.clip1[(q2-a3+256)&1023]
		}
	}
}

func loop26H(buf []byte, o, stride, n, thresh2, ithresh, hevT int) {
	t := fTab
	row := buf[o-4:]
	for j := 0; j < n; j++ {
		if j > 0 {
			row = row[stride:]
		}
		w := (*[8]byte)(row)
		p3, p2, p1, p0 := int(w[0]), int(w[1]), int(w[2]), int(w[3])
		q0, q1, q2, q3 := int(w[4]), int(w[5]), int(w[6]), int(w[7])
		if p0 == q0 && p1 == q1 {
			continue // the filter would leave this line unchanged
		}
		dp, dq := abs(p1-p0), abs(q1-q0)
		if 4*abs(p0-q0)+abs(p1-q1) > thresh2 ||
			max(abs(p3-p2), abs(p2-p1), dp, abs(q3-q2), abs(q2-q1), dq) > ithresh {
			continue
		}
		if max(dp, dq) > hevT {
			a := 3*(q0-p0) + sclip1(p1-q1)
			a1 := int(t.sclip2[uint8((a+4)>>3)])
			a2 := int(t.sclip2[uint8((a+3)>>3)])
			w[3] = t.clip1[(p0+a2+256)&1023]
			w[4] = t.clip1[(q0-a1+256)&1023]
		} else {
			a := sclip1(3*(q0-p0) + sclip1(p1-q1))
			a1 := (27*a + 63) >> 7
			a2 := (18*a + 63) >> 7
			a3 := (9*a + 63) >> 7
			w[1] = t.clip1[(p2+a3+256)&1023]
			w[2] = t.clip1[(p1+a2+256)&1023]
			w[3] = t.clip1[(p0+a1+256)&1023]
			w[4] = t.clip1[(q0-a1+256)&1023]
			w[5] = t.clip1[(q1-a2+256)&1023]
			w[6] = t.clip1[(q2-a3+256)&1023]
		}
	}
}

// loop24V filters an inner edge (up to 2 pixels modified on each side).
func loop24V(buf []byte, o, stride, n, thresh2, ithresh, hevT int) {
	t := fTab
	r0 := buf[o-4*stride:][:n]
	r1 := buf[o-3*stride:][:n]
	r2 := buf[o-2*stride:][:n]
	r3 := buf[o-stride:][:n]
	r4 := buf[o:][:n]
	r5 := buf[o+stride:][:n]
	r6 := buf[o+2*stride:][:n]
	r7 := buf[o+3*stride:][:n]
	for i := range r0 {
		p3, p2, p1, p0 := int(r0[i]), int(r1[i]), int(r2[i]), int(r3[i])
		q0, q1, q2, q3 := int(r4[i]), int(r5[i]), int(r6[i]), int(r7[i])
		if p0 == q0 && p1 == q1 {
			continue // the filter would leave this line unchanged
		}
		dp, dq := abs(p1-p0), abs(q1-q0)
		if 4*abs(p0-q0)+abs(p1-q1) > thresh2 ||
			max(abs(p3-p2), abs(p2-p1), dp, abs(q3-q2), abs(q2-q1), dq) > ithresh {
			continue
		}
		if max(dp, dq) > hevT {
			a := 3*(q0-p0) + sclip1(p1-q1)
			a1 := int(t.sclip2[uint8((a+4)>>3)])
			a2 := int(t.sclip2[uint8((a+3)>>3)])
			r3[i] = t.clip1[(p0+a2+256)&1023]
			r4[i] = t.clip1[(q0-a1+256)&1023]
		} else {
			a := 3 * (q0 - p0)
			a1 := int(t.sclip2[uint8((a+4)>>3)])
			a2 := int(t.sclip2[uint8((a+3)>>3)])
			a3 := (a1 + 1) >> 1
			r2[i] = t.clip1[(p1+a3+256)&1023]
			r3[i] = t.clip1[(p0+a2+256)&1023]
			r4[i] = t.clip1[(q0-a1+256)&1023]
			r5[i] = t.clip1[(q1-a3+256)&1023]
		}
	}
}

func loop24H(buf []byte, o, stride, n, thresh2, ithresh, hevT int) {
	t := fTab
	row := buf[o-4:]
	for j := 0; j < n; j++ {
		if j > 0 {
			row = row[stride:]
		}
		w := (*[8]byte)(row)
		p3, p2, p1, p0 := int(w[0]), int(w[1]), int(w[2]), int(w[3])
		q0, q1, q2, q3 := int(w[4]), int(w[5]), int(w[6]), int(w[7])
		if p0 == q0 && p1 == q1 {
			continue // the filter would leave this line unchanged
		}
		dp, dq := abs(p1-p0), abs(q1-q0)
		if 4*abs(p0-q0)+abs(p1-q1) > thresh2 ||
			max(abs(p3-p2), abs(p2-p1), dp, abs(q3-q2), abs(q2-q1), dq) > ithresh {
			continue
		}
		if max(dp, dq) > hevT {
			a := 3*(q0-p0) + sclip1(p1-q1)
			a1 := int(t.sclip2[uint8((a+4)>>3)])
			a2 := int(t.sclip2[uint8((a+3)>>3)])
			w[3] = t.clip1[(p0+a2+256)&1023]
			w[4] = t.clip1[(q0-a1+256)&1023]
		} else {
			a := 3 * (q0 - p0)
			a1 := int(t.sclip2[uint8((a+4)>>3)])
			a2 := int(t.sclip2[uint8((a+3)>>3)])
			a3 := (a1 + 1) >> 1
			w[2] = t.clip1[(p1+a3+256)&1023]
			w[3] = t.clip1[(p0+a2+256)&1023]
			w[4] = t.clip1[(q0-a1+256)&1023]
			w[5] = t.clip1[(q1-a3+256)&1023]
		}
	}
}

// filterMB applies the in-loop filter to the macroblock at column mbX of the
// current cache row (libwebp's DoFilter).
func (d *decoder) filterMB(mbX, mbY int, f fInfo) {
	limit := int(f.limit)
	if limit == 0 {
		return
	}
	ys := d.cacheYStride
	yo := d.cacheYOff + mbX*16
	cy := d.cacheY
	if d.filterType == 1 { // simple
		if mbX > 0 {
			simpleH(cy, yo, ys, 16, 2*(limit+4)+1)
		}
		if f.inner != 0 {
			for k := 4; k < 16; k += 4 {
				simpleH(cy, yo+k, ys, 16, 2*limit+1)
			}
		}
		if mbY > 0 {
			simpleV(cy, yo, ys, 16, 2*(limit+4)+1)
		}
		if f.inner != 0 {
			for k := 4; k < 16; k += 4 {
				simpleV(cy, yo+k*ys, ys, 16, 2*limit+1)
			}
		}
		return
	}
	// complex
	uvs := d.cacheUVStride
	uvo := d.cacheUVOff + mbX*8
	cu, cv := d.cacheU, d.cacheV
	il := int(f.ilevel)
	hevT := int(f.hevThresh)
	t2 := 2*(limit+4) + 1
	ti2 := 2*limit + 1
	if mbX > 0 {
		loop26H(cy, yo, ys, 16, t2, il, hevT)
		loop26H(cu, uvo, uvs, 8, t2, il, hevT)
		loop26H(cv, uvo, uvs, 8, t2, il, hevT)
	}
	if f.inner != 0 {
		loop24H(cy, yo+4, ys, 16, ti2, il, hevT)
		loop24H(cy, yo+8, ys, 16, ti2, il, hevT)
		loop24H(cy, yo+12, ys, 16, ti2, il, hevT)
		loop24H(cu, uvo+4, uvs, 8, ti2, il, hevT)
		loop24H(cv, uvo+4, uvs, 8, ti2, il, hevT)
	}
	if mbY > 0 {
		loop26V(cy, yo, ys, 16, t2, il, hevT)
		loop26V(cu, uvo, uvs, 8, t2, il, hevT)
		loop26V(cv, uvo, uvs, 8, t2, il, hevT)
	}
	if f.inner != 0 {
		loop24V(cy, yo+4*ys, ys, 16, ti2, il, hevT)
		loop24V(cy, yo+8*ys, ys, 16, ti2, il, hevT)
		loop24V(cy, yo+12*ys, ys, 16, ti2, il, hevT)
		loop24V(cu, uvo+4*uvs, uvs, 8, ti2, il, hevT)
		loop24V(cv, uvo+4*uvs, uvs, 8, ti2, il, hevT)
	}
}
