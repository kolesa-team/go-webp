package decoder

import (
	"image"
	"sync"
	"sync/atomic"
)

type ImagePool struct {
	poolMap map[int]*sync.Pool
	lock    *sync.Mutex
	Count   int64
}

func NewImagePool() *ImagePool {
	return &ImagePool{
		poolMap: make(map[int]*sync.Pool),
		lock:    &sync.Mutex{},
	}
}

func (n *ImagePool) Get(width, height int) *image.NRGBA {
	dimPool := n.getPool(width, height)

	img := dimPool.Get().(*image.NRGBA)
	img.Rect.Max.X = width
	img.Rect.Max.Y = height
	return img
}

func (n *ImagePool) getPool(width int, height int) *sync.Pool {
	dim := width * height

	n.lock.Lock()
	dimPool, ok := n.poolMap[dim]
	if !ok {
		atomic.AddInt64(&n.Count, 1)
		dimPool = &sync.Pool{
			New: func() interface{} {
				return image.NewNRGBA(image.Rect(0, 0, width, height))
			},
		}
		n.poolMap[dim] = dimPool
	}
	n.lock.Unlock()
	return dimPool
}

func (n *ImagePool) Put(img *image.NRGBA) {
	dimPool := n.getPool(img.Rect.Dx(), img.Rect.Dy())
	dimPool.Put(img)
}

type BufferPool struct {
	pool *sync.Pool // pointer because noCopy
}

func NewBufferPool() *BufferPool {
	return &BufferPool{
		pool: &sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, 1024)
			},
		},
	}
}

func (b *BufferPool) Get() []byte {
	return b.pool.Get().([]byte)
}

func (b *BufferPool) Put(buf []byte) {
	buf = buf[:0]
	b.pool.Put(buf)
}
