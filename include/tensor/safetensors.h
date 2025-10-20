#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <windows.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <cctype>
#include <thread>
struct TensorInfo
{
    std::string dtype;
    std::vector<size_t> shape;
    std::pair<size_t, size_t> data_offsets;
};

class MiniJson
{
public:
    MiniJson(const char *headerData, size_t size);
    MiniJson() = default;

    const std::vector<std::string> &keys() const { return key_order; }
    const TensorInfo *get(const std::string &key) const;
    const std::unordered_map<std::string, std::string> &getMetadata() const { return metadata; }

    void printMetadata() const;
    void print() const;

private:
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<std::string> key_order;

    // Utility parsing functions
    static void skipSpaces(std::istringstream &ss);
    static std::string readQuoted(std::istringstream &ss);
    static std::vector<size_t> readArray(std::istringstream &ss);
    static std::pair<size_t, size_t> readPair(std::istringstream &ss);
    static std::unordered_map<std::string, std::string> readStringMap(std::istringstream &ss);

    void parse(const std::string &json);
};

class Safetensor
{
public:
    Safetensor(const std::string &path, bool mmap = false);
    ~Safetensor();

    // Accessors
    const std::vector<std::string> &keys() const { return json.keys(); }
    const TensorInfo *getTensorInfo(const std::string &name) const { return json.get(name); }
    const std::unordered_map<std::string, std::string> &getMetadata() const { return json.getMetadata(); }

    void printHeader() const { json.print(); }

    // Returns pointer to tensor data (not owning)
    const uint8_t *tensorDataPtr(const std::string &key) const;
    size_t tensorByteSize(const std::string &key) const;

    static bool windows_advise(void *ptr, size_t size) noexcept;

private:
    MiniJson json;
    uint8_t *data;
    size_t data_size;
    bool is_mmap = false;
    HANDLE hFile_;
    HANDLE hMap_;

    // memory map variables

    void load(const std::string &path);
    void load_memory(const std::string &path);
    void load_mmap(const std::string &path);
    void cleanup_mmap();
};