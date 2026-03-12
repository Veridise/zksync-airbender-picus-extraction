CC=riscv64-elf-gcc
CFLAGS=-march=rv32im -mabi=ilp32 -nostdlib -O0 -nolibc -fno-builtin -ffreestanding -T memmap.ld
OBJCOPY=riscv64-elf-objcopy

.PHONY: clean all hello badread

all: hello badread 

hello: hello.bin hello.text
badread: badread.bin badread.text

%.elf: %.c rt.c start.s
	${CC} ${CFLAGS} -o $@ $^

%.bin: %.elf 
	${OBJCOPY} -O binary $^ $@ 

%.text: %.elf 
	${OBJCOPY} -O binary --only-section=.text $^ $@ 

%.asm: %.c 
	${CC} ${CFLAGS} -o $@ $^ -S

clean:
	rm -rf *.elf *.bin *.text *.asm
