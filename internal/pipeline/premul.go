package pipeline

// PremultipliedSink is implemented by sinks that alpha-premultiply the rows
// they receive before resampling them (Params.Premultiply). A decoder that
// can premultiply more cheaply than per pixel, such as a palette image that
// premultiplies its palette once, calls AcceptPremultiplied before its first
// Buffer call: when it returns true, the sink no longer premultiplies, and
// every row the decoder delivers must already be premultiplied exactly as
// resample.PremultiplyRGBA would do it.
type PremultipliedSink interface {
	AcceptPremultiplied() bool
}

// AcceptPremultiplied implements PremultipliedSink.
func (s *scaleSink) AcceptPremultiplied() bool {
	if !s.premult || s.scratch != nil || s.done {
		// Not premultiplying, or rows were already delivered.
		return false
	}
	s.preDone = true
	return true
}
