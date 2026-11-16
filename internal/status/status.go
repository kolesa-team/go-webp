// Package status carries libwebp-compatible VP8 status codes through Go errors,
// so the public API can keep reporting e.g. "VP8_STATUS_BITSTREAM_ERROR".
package status

import (
	"errors"

	"github.com/kolesa-team/go-webp/v2/utils"
)

// Error is an error tagged with a libwebp VP8StatusCode.
type Error struct {
	Code utils.VP8StatusCode
	Msg  string
}

func (e *Error) Error() string {
	if e.Msg == "" {
		return e.Code.String()
	}
	return e.Code.String() + ": " + e.Msg
}

// New returns an *Error with the given code and message.
func New(code utils.VP8StatusCode, msg string) error {
	return &Error{Code: code, Msg: msg}
}

// Bitstream is shorthand for New(utils.Vp8StatusBitstreamError, msg).
func Bitstream(msg string) error { return New(utils.Vp8StatusBitstreamError, msg) }

// NotEnoughData is shorthand for New(utils.Vp8StatusNotEnoughData, msg).
func NotEnoughData(msg string) error { return New(utils.Vp8StatusNotEnoughData, msg) }

// Unsupported is shorthand for New(utils.Vp8StatusUnsupportedFeature, msg).
func Unsupported(msg string) error { return New(utils.Vp8StatusUnsupportedFeature, msg) }

// InvalidParam is shorthand for New(utils.Vp8StatusInvalidParam, msg).
func InvalidParam(msg string) error { return New(utils.Vp8StatusInvalidParam, msg) }

// CodeOf returns the status code of err: Vp8StatusOk for nil, the tagged code
// for *Error (also when wrapped), and Vp8StatusBitstreamError otherwise.
func CodeOf(err error) utils.VP8StatusCode {
	if err == nil {
		return utils.Vp8StatusOk
	}
	var e *Error
	if errors.As(err, &e) {
		return e.Code
	}
	return utils.Vp8StatusBitstreamError
}
