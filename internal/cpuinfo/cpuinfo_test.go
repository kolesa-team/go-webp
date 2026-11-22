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

package cpuinfo

import (
	"runtime"
	"testing"
)

func TestLevelsConsistent(t *testing.T) {
	ls := Levels()
	if len(ls) == 0 || ls[0] != Go || ls[len(ls)-1] != Max {
		t.Fatalf("Levels() = %v with Max %v", ls, Max)
	}
	for i := 1; i < len(ls); i++ {
		if ls[i] != ls[i-1]+1 {
			t.Fatalf("Levels() = %v: not consecutive", ls)
		}
	}
	if X86.HasAVX512 && !X86.HasAVX2 || X86.HasAVX2 && !X86.HasSSE41 {
		t.Fatalf("inconsistent X86 flags %+v", X86)
	}
	if (X86.HasAVX512VBMI || X86.HasAVX512VBMI2 || X86.HasAVX512VNNI || X86.HasAVX512GFNI) && !X86.HasAVX512 {
		t.Fatalf("AVX-512 extensions without the AVX512 level: %+v", X86)
	}
	if runtime.GOARCH != "amd64" && Max != Go {
		t.Fatalf("Max = %v on %s", Max, runtime.GOARCH)
	}
	t.Logf("%s: Max %v, %+v", runtime.GOARCH, Max, X86)
}

func TestLevelString(t *testing.T) {
	for l, want := range map[Level]string{Go: "go", SSE41: "sse41", AVX2: "avx2", AVX512: "avx512", 9: "unknown"} {
		if got := l.String(); got != want {
			t.Errorf("Level(%d).String() = %q, want %q", int(l), got, want)
		}
	}
}
