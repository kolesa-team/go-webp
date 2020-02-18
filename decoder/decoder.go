package decoder

/*
#include <stdlib.h>
#include <webp/decode.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"image"
	"io"
	"io/ioutil"
	"unsafe"

	"github.com/kolesa-team/go-webp/utils"
)

type Decoder struct {
	data    []byte
	options *Options
	config  *C.WebPDecoderConfig
	dPtr    *C.uint8_t
	sPtr    C.size_t
}

func NewDecoder(r io.Reader, options *Options) (d *Decoder, err error) {
	var data []byte

	if data, err = ioutil.ReadAll(r); err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	d = &Decoder{data: data, options: options}

	if d.config, err = d.options.GetConfig(); err != nil {
		return nil, err
	}

	d.dPtr = (*C.uint8_t)(&d.data[0])
	d.sPtr = (C.size_t)(len(d.data))

	// получаем WebPBitstreamFeatures
	if status := d.parseFeatures(d.dPtr, d.sPtr); status != utils.Vp8StatusOk {
		return nil, errors.New(fmt.Sprintf("cannot fetch features: %s", status.String()))
	}

	return
}

func (d *Decoder) Decode() (image.Image, error) {
	// вписываем размеры итоговой картинки
	d.config.output.width, d.config.output.height = d.getOutputDimensions()
	// указываем что декодируем в RGBA
	d.config.output.colorspace = C.MODE_RGBA
	d.config.output.is_external_memory = 1

	img := image.NewNRGBA(image.Rectangle{Max: image.Point{
		X: int(d.config.output.width),
		Y: int(d.config.output.height),
	}})

	buff := (*C.WebPRGBABuffer)(unsafe.Pointer(&d.config.output.u[0]))
	buff.stride = C.int(img.Stride)
	buff.rgba = (*C.uint8_t)(&img.Pix[0])
	buff.size = (C.size_t)(len(img.Pix))

	if status := utils.VP8StatusCode(C.WebPDecode(d.dPtr, d.sPtr, d.config)); status != utils.Vp8StatusOk {
		return nil, errors.New(fmt.Sprintf("cannot decode picture: %s", status.String()))
	}

	return img, nil
}

func (d *Decoder) GetFeatures() utils.BitstreamFeatures {
	return utils.BitstreamFeatures{
		Width:        int(d.config.input.width),
		Height:       int(d.config.input.height),
		HasAlpha:     int(d.config.input.has_alpha) == 1,
		HasAnimation: int(d.config.input.has_animation) == 1,
		Format:       utils.FormatType(d.config.input.format),
	}
}

func (d *Decoder) parseFeatures(dataPtr *C.uint8_t, sizePtr C.size_t) utils.VP8StatusCode {
	return utils.VP8StatusCode(C.WebPGetFeatures(dataPtr, sizePtr, &d.config.input))
}

func (d *Decoder) getOutputDimensions() (width, height C.int) {
	width = d.config.input.width
	height = d.config.input.height

	if d.config.options.use_scaling > 0 {
		width = d.config.options.scaled_width
		height = d.config.options.scaled_height
	} else if d.config.options.use_cropping > 0 {
		width = d.config.options.crop_width
		height = d.config.options.crop_height
	}

	return
}
