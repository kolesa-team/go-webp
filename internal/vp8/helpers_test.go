package vp8

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"strconv"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

// riffInfo is what the tests need from a WebP container.
type riffInfo struct {
	vp8      []byte // VP8 chunk payload
	hasAlpha bool   // an ALPH chunk is present
}

// extractVP8 returns the VP8 chunk payload of a lossy WebP file.
func extractVP8(data []byte) (riffInfo, error) {
	var ri riffInfo
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return ri, errors.New("not a RIFF/WEBP file")
	}
	p := data[12:]
	for len(p) >= 8 {
		id := string(p[0:4])
		sz := int(binary.LittleEndian.Uint32(p[4:8]))
		p = p[8:]
		if sz > len(p) {
			return ri, errors.New("truncated chunk")
		}
		switch id {
		case "VP8 ":
			ri.vp8 = p[:sz]
			return ri, nil
		case "ALPH":
			ri.hasAlpha = true
		}
		p = p[sz+sz&1:]
	}
	return ri, errors.New("no VP8 chunk")
}

// readPAM parses a P7 RGB_ALPHA PAM file as written by dwebp -pam.
func readPAM(data []byte) (*image.NRGBA, error) {
	hdr := map[string]string{}
	r := bytes.NewBuffer(data)
	line, err := r.ReadString('\n')
	if err != nil || line != "P7\n" {
		return nil, fmt.Errorf("not a PAM file")
	}
	for {
		line, err = r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = line[:len(line)-1]
		if line == "ENDHDR" {
			break
		}
		if i := bytes.IndexByte([]byte(line), ' '); i > 0 {
			hdr[line[:i]] = line[i+1:]
		}
	}
	w, _ := strconv.Atoi(hdr["WIDTH"])
	h, _ := strconv.Atoi(hdr["HEIGHT"])
	if hdr["DEPTH"] != "4" || w <= 0 || h <= 0 || r.Len() < 4*w*h {
		return nil, fmt.Errorf("unsupported PAM header %v", hdr)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	copy(img.Pix, r.Bytes())
	return img, nil
}

// alphaPlane extracts the A channel of img.
func alphaPlane(img *image.NRGBA) []byte {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	a := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a[y*w+x] = img.Pix[y*img.Stride+4*x+3]
		}
	}
	return a
}

// decodeImage decodes a VP8 payload into a new NRGBA image.
func decodeImage(vp8 []byte, alpha []byte, opts *Options) (*image.NRGBA, error) {
	w, h, err := DecodeHeader(vp8)
	if err != nil {
		return nil, err
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	if err := Decode(vp8, alpha, opts, pipeline.NewDirect(img)); err != nil {
		return nil, err
	}
	return img, nil
}

// firstDiff describes the first mismatching pixel of two images.
func firstDiff(a, b *image.NRGBA) string {
	if a.Rect != b.Rect {
		return fmt.Sprintf("size %v vs %v", a.Rect, b.Rect)
	}
	n := 0
	first := ""
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			if first == "" {
				px := i / 4
				first = fmt.Sprintf("first at x=%d y=%d ch=%d got %d want %d",
					px%a.Rect.Dx(), px/a.Rect.Dx(), i%4, a.Pix[i], b.Pix[i])
			}
			n++
		}
	}
	if n == 0 {
		return ""
	}
	return fmt.Sprintf("%d bytes differ; %s", n, first)
}
