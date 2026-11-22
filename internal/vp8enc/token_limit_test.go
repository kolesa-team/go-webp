// The MIT License (MIT)
//
// Copyright (c) 2019 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package vp8enc

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/kolesa-team/go-webp/v2/internal/status"
	"github.com/kolesa-team/go-webp/v2/internal/workers"
	"github.com/kolesa-team/go-webp/v2/utils"
)

// setMaxTokensForTest lowers maxTokens for the rest of the test.
func setMaxTokensForTest(tb testing.TB, n int) {
	old := maxTokens
	maxTokens = n
	tb.Cleanup(func() { maxTokens = old })
}

// encodeWithMaxTokens encodes img with maxTokens set to limit.
func encodeWithMaxTokens(tb testing.TB, img *Picture, cfg Config, limit int) ([]byte, error) {
	tb.Helper()
	old := maxTokens
	maxTokens = limit
	defer func() { maxTokens = old }()
	return Encode(img, &cfg)
}

// minTokenLimit returns the smallest maxTokens with which cfg encodes pic:
// the token count of the pass that records the most tokens.
func minTokenLimit(tb testing.TB, pic *Picture, cfg Config) int {
	tb.Helper()
	lo, hi := 0, 1<<26 // Encode fails with lo and succeeds with hi
	if _, err := encodeWithMaxTokens(tb, pic, cfg, hi); err != nil {
		tb.Fatalf("encode with %d tokens: %v", hi, err)
	}
	for hi-lo > 1 {
		mid := lo + (hi-lo)/2
		_, err := encodeWithMaxTokens(tb, pic, cfg, mid)
		switch {
		case err == nil:
			hi = mid
		case errors.Is(err, ErrTooManyTokens):
			lo = mid
		default:
			tb.Fatalf("encode with %d tokens: %v", mid, err)
		}
	}
	return hi
}

// TestTokenLimit checks that a coding pass recording more than maxTokens
// tokens fails with ErrTooManyTokens instead of overflowing the int32 token
// offsets (which made the encoder index its token buffer out of range),
// sequentially and as a wavefront with every helper budget, and that the
// limit itself still encodes, to the same bytes.
func TestTokenLimit(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(0))
	img := synthImage(320, 240, 3, 7) // 20x15 macroblocks, noise
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	for _, tc := range []struct {
		name string
		mod  func(*Config)
	}{
		{"m4", func(c *Config) {}},
		{"m0", func(c *Config) { c.Method = 0 }},
		{"m6-q100", func(c *Config) { c.Method, c.Quality = 6, 100 }},
		{"target-size", func(c *Config) { c.Pass, c.TargetSize = 4, 20000 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tc.mod(&cfg)
			workers.SetLimitForTest(0)
			limit := minTokenLimit(t, pic, cfg)
			want, err := Encode(pic, &cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, threads := range []bool{false, true} {
				cfg.ThreadLevel = threads
				for _, budget := range []int{0, 1, 4, 16} { // 0: the default, GOMAXPROCS-1
					if !threads && budget > 0 {
						continue
					}
					workers.SetLimitForTest(budget)
					name := fmt.Sprintf("threads=%v budget=%d", threads, budget)
					got, err := encodeWithMaxTokens(t, pic, cfg, limit)
					if err != nil {
						t.Fatalf("%s: limit %d: %v", name, limit, err)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("%s: limit %d changed the output", name, limit)
					}
					for _, l := range []int{limit - 1, limit / 2, 1, 0} {
						_, err := encodeWithMaxTokens(t, pic, cfg, l)
						if !errors.Is(err, ErrTooManyTokens) {
							t.Fatalf("%s: limit %d (needs %d): got %v, want ErrTooManyTokens", name, l, limit, err)
						}
						if h := workers.InUse(); h != 0 {
							t.Fatalf("%s: limit %d: %d helpers running after Encode returned", name, l, h)
						}
					}
				}
			}
		})
	}
}

// TestTokenLimitThenEncode checks that a pass stopped by the token limit
// leaves nothing behind: the pooled encoder and wavefront state encode the
// next picture to the same bytes as before.
func TestTokenLimitThenEncode(t *testing.T) {
	forceWavefrontForTest(t)
	defer workers.SetLimitForTest(workers.SetLimitForTest(4))
	img := synthImage(320, 240, 3, 11)
	pic := FromNRGBA(img, false, false)
	defer pic.Release()
	cfg := DefaultConfig()
	cfg.ThreadLevel = true
	want, err := Encode(pic, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 20 {
		if _, err := encodeWithMaxTokens(t, pic, cfg, 1000*(i+1)); !errors.Is(err, ErrTooManyTokens) {
			t.Fatalf("limit %d: got %v, want ErrTooManyTokens", 1000*(i+1), err)
		}
		got, err := Encode(pic, &cfg)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("encode after a stopped pass (limit %d) changed the output", 1000*(i+1))
		}
	}
}

// TestErrTooManyTokensStatus checks that the error reports libwebp's status
// for a token buffer that cannot grow.
func TestErrTooManyTokensStatus(t *testing.T) {
	setMaxTokensForTest(t, 0)
	pic := FromNRGBA(synthImage(32, 32, 0, 1), false, false)
	defer pic.Release()
	_, err := Encode(pic, nil)
	if !errors.Is(err, ErrTooManyTokens) {
		t.Fatalf("got %v, want ErrTooManyTokens", err)
	}
	if got := status.CodeOf(err); got != utils.Vp8StatusOutOfMemory {
		t.Fatalf("status %v, want %v", got, utils.Vp8StatusOutOfMemory)
	}
}
