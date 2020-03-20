package encoder

/*
#cgo LDFLAGS: -lwebp
#include <stdlib.h>
#include <webp/encode.h>
static uint8_t* encodeNRBBA(WebPConfig* config, const uint8_t* rgba, int width, int height, int stride, size_t* output_size) {
	WebPPicture pic;
	WebPMemoryWriter wrt;
	int ok;

	if (!WebPPictureInit(&pic)) {
		return NULL;
	}

	pic.use_argb = 1;
	pic.width = width;
	pic.height = height;
	pic.writer = WebPMemoryWrite;
	pic.custom_ptr = &wrt;
	WebPMemoryWriterInit(&wrt);

	ok = WebPPictureImportRGBA(&pic, rgba, stride) && WebPEncode(config, &pic);
	WebPPictureFree(&pic);

	if (!ok) {
		WebPMemoryWriterClear(&wrt);
		return NULL;
	}

	*output_size = wrt.size;
	return wrt.mem;
}
*/
import "C"
import (
	"errors"
	"image"
	"image/draw"
	"io"
	"unsafe"
)

type (
	Encoder struct {
		options *Options
		config  *C.WebPConfig
		img     *image.NRGBA
	}
)

func NewEncoder(src image.Image, options *Options) (e *Encoder, err error) {
	var config *C.WebPConfig

	if config, err = options.GetConfig(); err != nil {
		return nil, err
	}

	e = &Encoder{options: options, config: config}

	switch v := src.(type) {
	case *image.NRGBA:
		e.img = v
	default:
		e.img = e.convertToNRGBA(src)
	}

	return
}

func (e *Encoder) Encode(w io.Writer) error {
	var size C.size_t

	output := C.encodeNRBBA(
		e.config,
		(*C.uint8_t)(&e.img.Pix[0]),
		C.int(e.img.Rect.Max.X),
		C.int(e.img.Rect.Max.Y),
		C.int(e.img.Stride),
		&size,
	)

	if output == nil || size == 0 {
		return errors.New("cannot encode webppicture")
	}
	defer C.free(unsafe.Pointer(output))

	_, err := w.Write(((*[1 << 30]byte)(unsafe.Pointer(output)))[0:int(size):int(size)])

	return err
}

func (e *Encoder) convertToNRGBA(src image.Image) (dst *image.NRGBA) {
	dst = image.NewNRGBA(src.Bounds())
	draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)

	return
}
