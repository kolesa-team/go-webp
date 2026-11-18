package suite

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"os"
	"runtime"
	"sync"
	"time"
)

// RSSMain is the main function of the peak-memory tools (../v1/rss,
// ../v2/rss). Each run measures one workload in a fresh process and prints
// one line: the wall time of the workload, the resident memory before it
// and the peak resident memory of the whole process (getrusage maxrss).
//
// Modes:
//   - enc12: one lossy q75 encode of a 4000x3000 *image.YCbCr made by
//     tiling source.jpg (-threads: Options.ThreadLevel);
//   - enc12ll: -r lossless level 6 encodes in a row of a 4000x3000
//     *image.YCbCr upscaled from source.jpg (a photo-like 12 MP picture);
//   - gen12ll: write that 12 MP lossless picture to -out (input for dec12);
//   - dec12: one decode of -file;
//   - conc: -n goroutines decoding -file -r times each (-into: DecodeInto
//     with one output buffer per goroutine).
func RSSMain(c Codec) {
	mode := flag.String("mode", "", "enc12, enc12ll, gen12ll, dec12 or conc")
	file := flag.String("file", "", "input picture (dec12, conc)")
	out := flag.String("out", "", "output file (gen12ll)")
	r := flag.Int("r", 1, "encodes in a row (enc12ll), decodes per goroutine (conc)")
	n := flag.Int("n", 15, "goroutines (conc)")
	into := flag.Bool("into", false, "conc: DecodeInto with one buffer per goroutine")
	threads := flag.Bool("threads", false, "encodes: Options.ThreadLevel")
	flag.Parse()

	fail := func(err error) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var base int64
	var size int
	var t0 time.Time
	start := func() {
		runtime.GC()
		base = peakRSS()
		t0 = time.Now()
	}
	switch *mode {
	case "enc12":
		img, err := tiledYCbCr(4000, 3000)
		if err != nil {
			fail(err)
		}
		start()
		if size, err = encodedSize(c, img, EncodeCase{Level: 75, Method: 4}, *threads); err != nil {
			fail(err)
		}
	case "enc12ll", "gen12ll":
		img, err := upscaledYCbCr(4000, 3000)
		if err != nil {
			fail(err)
		}
		ec := EncodeCase{Lossless: true, Level: 6}
		if *mode == "gen12ll" {
			var buf bytes.Buffer
			if err := c.Encode(&buf, img, ec, true); err != nil {
				fail(err)
			}
			if err := os.WriteFile(*out, buf.Bytes(), 0o644); err != nil {
				fail(err)
			}
			fmt.Printf("wrote %s (%d bytes)\n", *out, buf.Len())
			return
		}
		start()
		for i := 0; i < *r; i++ {
			s, err := encodedSize(c, img, ec, *threads)
			if err != nil {
				fail(err)
			}
			if i == 0 {
				size = s
			}
		}
	case "dec12":
		data, err := os.ReadFile(*file)
		if err != nil {
			fail(err)
		}
		start()
		if _, err := c.Decode(data); err != nil {
			fail(err)
		}
	case "conc":
		data, err := os.ReadFile(*file)
		if err != nil {
			fail(err)
		}
		start()
		var wg sync.WaitGroup
		errs := make(chan error, *n)
		for g := 0; g < *n; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				var dst *image.NRGBA
				for i := 0; i < *r; i++ {
					var err error
					if *into {
						dst, err = c.DecodeInto(data, dst)
					} else {
						_, err = c.Decode(data)
					}
					if err != nil {
						errs <- err
						return
					}
				}
			}()
		}
		wg.Wait()
		close(errs)
		if err := <-errs; err != nil {
			fail(err)
		}
	default:
		fail(fmt.Errorf("unknown -mode %q", *mode))
	}
	fmt.Printf("wall_ms=%.0f base_mb=%.1f peak_mb=%.1f", float64(time.Since(t0).Microseconds())/1000, float64(base)/1e6, float64(peakRSS())/1e6)
	if size > 0 {
		fmt.Printf(" bytes=%d", size)
	}
	fmt.Println()
}

func sourceYCbCr() (*image.YCbCr, error) {
	img, err := LoadJPEG()
	if err != nil {
		return nil, err
	}
	s, ok := img.(*image.YCbCr)
	if !ok {
		return nil, fmt.Errorf("source.jpg: want an *image.YCbCr, got %T", img)
	}
	return s, nil
}

// chromaDiv returns the horizontal and vertical chroma subsampling factors.
func chromaDiv(r image.YCbCrSubsampleRatio) (int, int) {
	switch r {
	case image.YCbCrSubsampleRatio422:
		return 2, 1
	case image.YCbCrSubsampleRatio420:
		return 2, 2
	case image.YCbCrSubsampleRatio440:
		return 1, 2
	case image.YCbCrSubsampleRatio411:
		return 4, 1
	case image.YCbCrSubsampleRatio410:
		return 4, 2
	}
	return 1, 1
}

// tiledYCbCr tiles source.jpg into a w x h 4:2:0 picture, like a camera
// JPEG (the chroma of each 2x2 block is the source's at its top-left).
func tiledYCbCr(w, h int) (*image.YCbCr, error) {
	s, err := sourceYCbCr()
	if err != nil {
		return nil, err
	}
	dst := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
	sw, sh := s.Rect.Dx(), s.Rect.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.Y[dst.YOffset(x, y)] = s.Y[s.YOffset(x%sw, y%sh)]
		}
	}
	for y := 0; y < h; y += 2 {
		for x := 0; x < w; x += 2 {
			do, so := dst.COffset(x, y), s.COffset(x%sw, y%sh)
			dst.Cb[do], dst.Cr[do] = s.Cb[so], s.Cr[so]
		}
	}
	return dst, nil
}

// upscaledYCbCr scales source.jpg bilinearly to a w x h 4:2:0 picture
// (tiling would make a lossless encode unrealistically cheap through LZ77
// matches).
func upscaledYCbCr(w, h int) (*image.YCbCr, error) {
	s, err := sourceYCbCr()
	if err != nil {
		return nil, err
	}
	dst := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
	// sample interpolates the pw x ph plane p at (fx, fy).
	sample := func(p []byte, stride, pw, ph int, fx, fy float64) byte {
		x0, y0 := int(fx), int(fy)
		x1, y1 := min(x0+1, pw-1), min(y0+1, ph-1)
		ax, ay := fx-float64(x0), fy-float64(y0)
		v := (1-ay)*((1-ax)*float64(p[y0*stride+x0])+ax*float64(p[y0*stride+x1])) +
			ay*((1-ax)*float64(p[y1*stride+x0])+ax*float64(p[y1*stride+x1]))
		return byte(v + 0.5)
	}
	sw, sh := s.Rect.Dx(), s.Rect.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.Y[y*dst.YStride+x] = sample(s.Y, s.YStride, sw, sh, float64(x)*float64(sw-1)/float64(w-1), float64(y)*float64(sh-1)/float64(h-1))
		}
	}
	hd, vd := chromaDiv(s.SubsampleRatio)
	cw, ch := (sw+hd-1)/hd, (sh+vd-1)/vd
	for y := 0; y < h/2; y++ {
		for x := 0; x < w/2; x++ {
			fx, fy := float64(x)*float64(cw-1)/float64(w/2-1), float64(y)*float64(ch-1)/float64(h/2-1)
			dst.Cb[y*dst.CStride+x] = sample(s.Cb, s.CStride, cw, ch, fx, fy)
			dst.Cr[y*dst.CStride+x] = sample(s.Cr, s.CStride, cw, ch, fx, fy)
		}
	}
	return dst, nil
}
