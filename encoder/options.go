package encoder

/*
#cgo LDFLAGS: -lwebp
#include <webp/encode.h>
*/
import "C"
import "errors"

//noinspection GoUnusedConst
const (
	HintDefault ImageHint = iota
	HintPicture
	HintPhoto
	HintGraph
	HintLast
)

//noinspection GoUnusedConst
const (
	PresetDefault EncodingPreset = iota
	PresetPicture
	PresetPhoto
	PresetDrawing
	PresetIcon
	PresetText
)

type (
	ImageHint      int
	EncodingPreset int
	Options        struct {
		config *C.WebPConfig

		Lossless         bool
		Quality          float32
		Method           int
		ImageHint        ImageHint
		TargetSize       int
		TargetPsnr       float32
		Segments         int
		SnsStrength      int
		FilterStrength   int
		FilterSharpness  int
		FilterType       int
		Autofilter       bool
		AlphaCompression int
		AlphaFiltering   int
		alphaQuality     int
		Pass             int
		//QMin             int
		//QMax             int
		ShowCompressed  bool
		Preprocessing   int
		Partitions      int
		PartitionLimit  int
		EmulateJpegSize bool
		ThreadLevel     bool
		LowMemory       bool
		NearLossless    int
		Exact           int
		UseDeltaPalette bool
		UseSharpYuv     bool
	}
)

func NewLossyEncoderOptions(preset EncodingPreset, quality float32) (options *Options, err error) {
	options = &Options{
		config: &C.WebPConfig{},
	}

	if C.WebPConfigPreset(options.config, C.WebPPreset(preset), C.float(quality)) == 0 {
		return nil, errors.New("cannot init encoder config")
	}

	options.sync()

	return
}

func NewLosslessEncoderOptions(preset EncodingPreset, level int) (options *Options, err error) {
	if options, err = NewLossyEncoderOptions(preset, 0); err != nil {
		return
	}
	if C.WebPConfigLosslessPreset(options.config, C.int(level)) == 0 {
		return nil, errors.New("cannot init lossless preset")
	}

	options.sync()

	return
}

func (o *Options) sync() {
	o.Lossless = o.config.lossless == 1
	o.Quality = float32(o.config.quality)
	o.Method = int(o.config.method)
	o.ImageHint = ImageHint(o.config.image_hint)
	o.TargetSize = int(o.config.target_size)
	o.TargetPsnr = float32(o.config.target_PSNR)
	o.Segments = int(o.config.segments)
	o.SnsStrength = int(o.config.sns_strength)
	o.FilterStrength = int(o.config.filter_strength)
	o.FilterSharpness = int(o.config.filter_sharpness)
	o.FilterType = int(o.config.filter_type)
	o.Autofilter = o.config.autofilter == 1
	o.AlphaCompression = int(o.config.alpha_compression)
	o.AlphaFiltering = int(o.config.alpha_filtering)
	o.alphaQuality = int(o.config.alpha_quality)
	o.Pass = int(o.config.pass)
	//o.QMin = int(o.config.qmin)
	//o.QMax = int(o.config.qmax)
	o.ShowCompressed = o.config.show_compressed == 1
	o.Preprocessing = int(o.config.preprocessing)
	o.Partitions = int(o.config.partitions)
	o.PartitionLimit = int(o.config.partition_limit)
	o.EmulateJpegSize = o.config.emulate_jpeg_size == 1
	o.ThreadLevel = o.config.thread_level == 1
	o.LowMemory = o.config.low_memory == 1
	o.NearLossless = int(o.config.near_lossless)
	o.Exact = int(o.config.exact)
	o.UseDeltaPalette = o.config.use_delta_palette == 1
	o.UseSharpYuv = o.config.use_sharp_yuv == 1
}

func (o *Options) boolToCInt(expression bool) (result C.int) {
	result = 0

	if expression {
		result = 1
	}

	return
}

func (o *Options) GetConfig() (*C.WebPConfig, error) {
	o.config.lossless = o.boolToCInt(o.Lossless)
	o.config.quality = C.float(o.Quality)
	o.config.method = C.int(o.Method)
	o.config.image_hint = C.WebPImageHint(o.ImageHint)
	o.config.target_size = C.int(o.TargetSize)
	o.config.target_PSNR = C.float(o.TargetPsnr)
	o.config.segments = C.int(o.Segments)
	o.config.sns_strength = C.int(o.SnsStrength)
	o.config.filter_strength = C.int(o.FilterStrength)
	o.config.filter_sharpness = C.int(o.FilterSharpness)
	o.config.filter_type = C.int(o.FilterType)
	o.config.autofilter = o.boolToCInt(o.Autofilter)
	o.config.alpha_compression = C.int(o.AlphaCompression)
	o.config.alpha_filtering = C.int(o.AlphaFiltering)
	o.config.alpha_quality = C.int(o.alphaQuality)
	o.config.pass = C.int(o.Pass)
	//o.config.qmin = C.int(o.QMin)
	//o.config.qmax = C.int(o.QMax)
	o.config.show_compressed = o.boolToCInt(o.ShowCompressed)
	o.config.preprocessing = C.int(o.Preprocessing)
	o.config.partitions = C.int(o.Partitions)
	o.config.partition_limit = C.int(o.PartitionLimit)
	o.config.emulate_jpeg_size = o.boolToCInt(o.EmulateJpegSize)
	o.config.thread_level = o.boolToCInt(o.ThreadLevel)
	o.config.low_memory = o.boolToCInt(o.LowMemory)
	o.config.near_lossless = C.int(o.NearLossless)
	o.config.exact = C.int(o.Exact)
	o.config.use_delta_palette = o.boolToCInt(o.UseDeltaPalette)
	o.config.use_sharp_yuv = o.boolToCInt(o.UseSharpYuv)

	if C.WebPValidateConfig(o.config) == 0 {
		return nil, errors.New("cannot validate config")
	}

	return o.config, nil
}
