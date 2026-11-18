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
