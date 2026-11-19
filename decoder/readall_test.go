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

package decoder

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"runtime"
	"testing"
	"testing/iotest"
)

// TestReadAllPresized checks that readers of unknown size are read correctly
// into a buffer sized from the RIFF header (one allocation for files up to
// initialReadSize), including the bufio.Reader that image.Decode wraps
// around its input.
func TestReadAllPresized(t *testing.T) {
	data, err := os.ReadFile("../test_data/images/m4_q75.webp")
	if err != nil {
		t.Fatal(err)
	}
	for name, r := range map[string]io.Reader{
		"plain":   struct{ io.Reader }{bytes.NewReader(data)},
		"bufio":   bufio.NewReader(bytes.NewReader(data)),
		"onebyte": iotest.OneByteReader(bytes.NewReader(data)),
	} {
		b, err := readAll(r)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !bytes.Equal(b, data) {
			t.Fatalf("%s: data differs", name)
		}
		if cap(b) > 2*len(data)+1 {
			t.Errorf("%s: cap %d, want <= %d", name, cap(b), 2*len(data)+1)
		}
	}
	// Short and non-WebP inputs are read as is.
	for _, in := range [][]byte{nil, []byte("RIFF"), []byte("not a webp file at all")} {
		b, err := readAll(struct{ io.Reader }{bytes.NewReader(in)})
		if err != nil || !bytes.Equal(b, in) {
			t.Errorf("readAll(%q) = %q, %v", in, b, err)
		}
	}
}

// TestReadAllLyingHeader checks that a tiny input whose RIFF header claims a
// huge size does not trigger a large allocation (it used to cost 64 MB).
func TestReadAllLyingHeader(t *testing.T) {
	in := []byte("RIFF\xff\xff\xff\xffWEBPVP8 ")
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < 10; i++ {
		b, err := readAll(bufio.NewReader(bytes.NewReader(in)))
		if err != nil || !bytes.Equal(b, in) {
			t.Fatalf("readAll = %q, %v", b, err)
		}
		if cap(b) > 1<<20 {
			t.Fatalf("cap %d for a %d-byte input", cap(b), len(in))
		}
	}
	runtime.ReadMemStats(&after)
	if d := after.TotalAlloc - before.TotalAlloc; d > 10<<20 {
		t.Fatalf("10 reads of a 16-byte input allocated %d bytes", d)
	}
	// NewDecoder through bufio (the image.Decode path) must stay small too.
	runtime.ReadMemStats(&before)
	_, _ = NewDecoder(bufio.NewReader(bytes.NewReader(in)), &Options{})
	runtime.ReadMemStats(&after)
	if d := after.TotalAlloc - before.TotalAlloc; d > 1<<20 {
		t.Fatalf("NewDecoder allocated %d bytes for a 16-byte input", d)
	}
}

// TestReadAllLargeStream checks that a truthful header still reads a large
// stream correctly with a bounded number of buffer growths.
func TestReadAllLargeStream(t *testing.T) {
	data := make([]byte, 5<<20)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WEBP")
	for i := 12; i < len(data); i++ {
		data[i] = byte(i * 7)
	}
	b, err := readAll(bufio.NewReader(bytes.NewReader(data)))
	if err != nil || !bytes.Equal(b, data) {
		t.Fatalf("readAll mismatch: %v", err)
	}
	if cap(b) != len(data)+1 {
		t.Errorf("cap %d, want %d", cap(b), len(data)+1)
	}
	// Header understating the size: still reads everything.
	binary.LittleEndian.PutUint32(data[4:], 100)
	b, err = readAll(bufio.NewReader(bytes.NewReader(data)))
	if err != nil || !bytes.Equal(b, data) {
		t.Fatalf("readAll (short header) mismatch: %v", err)
	}
}
