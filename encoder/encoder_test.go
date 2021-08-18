package encoder

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"image"
	"io/ioutil"
	"testing"
)

func TestNewEncoder(t *testing.T) {
	t.Run("encode nrgba image with lossy preset", func(t *testing.T) {
		expected := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rectangle{
			Max: image.Point{
				X: 100,
				Y: 150,
			},
		})

		options, err := NewLossyEncoderOptions(PresetDefault, 0.75)
		require.NoError(t, err)

		e, err := NewEncoder(img, options)
		require.NoError(t, err)

		err = e.Encode(expected)
		require.NoError(t, err)

		actual, err := ioutil.ReadFile("../test_data/images/100x150_lossy.webp")
		require.NoError(t, err)

		assert.Equal(t, actual, expected.Bytes())
	})
	t.Run("encode nrgba image with lossless preset", func(t *testing.T) {
		actuall := &bytes.Buffer{}
		img := image.NewNRGBA(image.Rectangle{
			Max: image.Point{
				X: 100,
				Y: 150,
			},
		})

		options, err := NewLosslessEncoderOptions(PresetDefault, 1)
		require.NoError(t, err)

		e, err := NewEncoder(img, options)
		require.NoError(t, err)

		err = e.Encode(actuall)
		require.NoError(t, err)

		expected, err := ioutil.ReadFile("../test_data/images/100x150_lossless.webp")
		require.NoError(t, err)

		assert.Equal(t, expected, actuall.Bytes())
	})
}
