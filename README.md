# go-webp
Golang Webp library for encoding and decoding, with C binding for Google [libwebp](https://developers.google.com/speed/webp/docs/api)

## Install libwebp
#### MacOS:
```bash
brew install webp
```
#### Linux:
```bash
sudo apt-get update
sudo apt-get install libwebp-dev
```

## Install go-webp
`go get -u github.com/kolesa-team/go-webp`

## Example 
#### Before run
```bash
export CGO_CFLAGS="-I /usr/local/opt/webp/include"
export CGO_LDFLAGS="-L /usr/local/opt/webp/lib -lwebp"
export LD_LIBRARY_PATH=/usr/local/opt/webp/lib
```

#### Decode:
```go
package main

import (
	"image/jpeg"
	"log"
	"os"

	"github.com/kolesa-team/go-webp/decoder"
	"github.com/kolesa-team/go-webp/webp"
)

func main() {
	file, err := os.Open("test_data/images/m4_q75.webp")
	if err != nil {
		log.Fatalln(err)
	}

	output, err := os.Create("example/output_decode.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer output.Close()

	img, err := webp.Decode(file, &decoder.Options{})
	if err != nil {
		log.Fatalln(err)
	}

	if err = jpeg.Encode(output, img, &jpeg.Options{Quality:75}); err != nil {
		log.Fatalln(err)
	}
}
```

```bash
go run example/decode/main.go
```

#### Encode
```go
package main

import (
	"github.com/kolesa-team/go-webp/encoder"
	"github.com/kolesa-team/go-webp/webp"
	"image/jpeg"
	"log"
	"os"
)

func main() {
	file, err := os.Open("test_data/images/source.jpg")
	if err != nil {
		log.Fatalln(err)
	}

	img, err := jpeg.Decode(file)
	if err != nil {
		log.Fatalln(err)
	}

	output, err := os.Create("example/output_decode.webp")
	if err != nil {
		log.Fatal(err)
	}
	defer output.Close()

	options, err := encoder.NewLossyEncoderOptions(encoder.PresetDefault, 75)
	if err != nil {
		log.Fatalln(err)
	}

	if err := webp.Encode(output, img, options); err != nil {
		log.Fatalln(err)
	}
}
```
```bash
go run example/encode/main.go
```

## TODO
- return aux stats
- container api
- incremental decoding

## License
BSD licensed. See the LICENSE file for details.

