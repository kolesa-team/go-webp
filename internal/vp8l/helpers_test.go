package vp8l

import (
	"encoding/binary"
	"image"

	"github.com/kolesa-team/go-webp/v2/internal/pipeline"
)

// findChunk returns the payload of the first RIFF chunk with the given id.
func findChunk(file []byte, id string) []byte {
	if len(file) < 12 || string(file[:4]) != "RIFF" || string(file[8:12]) != "WEBP" {
		return nil
	}
	p := 12
	for p+8 <= len(file) {
		n := int(binary.LittleEndian.Uint32(file[p+4:]))
		if p+8+n > len(file) {
			return nil
		}
		if string(file[p:p+4]) == id {
			return file[p+8 : p+8+n]
		}
		p += 8 + n + n&1
	}
	return nil
}

// decodeVP8L decodes a VP8L payload to an NRGBA image via the Direct sink.
func decodeVP8L(payload []byte) (*image.NRGBA, error) {
	w, h, _, err := DecodeHeader(payload)
	if err != nil {
		return nil, err
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	if err := Decode(payload, pipeline.NewDirect(img)); err != nil {
		return nil, err
	}
	return img, nil
}

// unfilterAlpha undoes the ALPH prediction filter (1 horizontal,
// 2 vertical, 3 gradient) in place, as libwebp does.
func unfilterAlpha(a []byte, w, h, filter int) {
	for y := 0; y < h; y++ {
		row := a[y*w : (y+1)*w]
		var prev []byte
		if y > 0 {
			prev = a[(y-1)*w : y*w]
		}
		if filter == 0 {
			return
		}
		if prev == nil || filter == 1 {
			pred := byte(0)
			if prev != nil {
				pred = prev[0]
			}
			for i := range row {
				row[i] += pred
				pred = row[i]
			}
			continue
		}
		if filter == 2 {
			for i := range row {
				row[i] += prev[i]
			}
			continue
		}
		top := int(prev[0])
		topLeft, left := top, top
		for i := range row {
			top = int(prev[i])
			g := left + top - topLeft
			if g < 0 {
				g = 0
			} else if g > 255 {
				g = 255
			}
			left = int(row[i]+byte(g)) & 0xff
			topLeft = top
			row[i] = byte(left)
		}
	}
}

// alphaFromFile extracts the ALPH header byte, its payload and the canvas
// size of a VP8X file.
func alphaFromFile(file []byte) (hdr byte, payload []byte, w, h int, ok bool) {
	x := findChunk(file, "VP8X")
	a := findChunk(file, "ALPH")
	if len(x) < 10 || len(a) < 1 {
		return 0, nil, 0, 0, false
	}
	w = int(x[4]) | int(x[5])<<8 | int(x[6])<<16 + 1
	h = int(x[7]) | int(x[8])<<8 | int(x[9])<<16 + 1
	return a[0], a[1:], w, h, true
}
